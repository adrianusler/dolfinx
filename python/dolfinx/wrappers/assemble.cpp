// Copyright (C) 2017-2021 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "array.h"
#include "pycoeff.h"
#include <array>
#include <cstdint>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/assembler.h>
#include <dolfinx/fem/discreteoperators.h>
#include <dolfinx/fem/sparsitybuild.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/mesh/Mesh.h>
#include <memory>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <span>
#include <string>
#include <utility>

namespace py = pybind11;

namespace
{

// Declare assembler function that have multiple scalar types
template <typename T, typename U>
void declare_discrete_operators(py::module& m)
{
  m.def(
      "discrete_gradient",
      [](const dolfinx::fem::FunctionSpace<U>& V0,
         const dolfinx::fem::FunctionSpace<U>& V1)
      {
        assert(V0.mesh());
        auto mesh = V0.mesh();
        assert(V1.mesh());
        assert(mesh == V1.mesh());
        MPI_Comm comm = mesh->comm();

        auto dofmap0 = V0.dofmap();
        assert(dofmap0);
        auto dofmap1 = V1.dofmap();
        assert(dofmap1);

        // Create and build  sparsity pattern
        assert(dofmap0->index_map);
        assert(dofmap1->index_map);
        dolfinx::la::SparsityPattern sp(
            comm, {dofmap1->index_map, dofmap0->index_map},
            {dofmap1->index_map_bs(), dofmap0->index_map_bs()});

        int tdim = mesh->topology()->dim();
        auto map = mesh->topology()->index_map(tdim);
        assert(map);
        std::vector<std::int32_t> c(map->size_local(), 0);
        std::iota(c.begin(), c.end(), 0);
        dolfinx::fem::sparsitybuild::cells(sp, c, {*dofmap1, *dofmap0});
        sp.finalize();

        // Build operator
        dolfinx::la::MatrixCSR<T> A(sp);
        dolfinx::fem::discrete_gradient<T, U>(
            *V0.mesh()->topology_mutable(), {*V0.element(), *V0.dofmap()},
            {*V1.element(), *V1.dofmap()}, A.mat_set_values());
        return A;
      },
      py::arg("V0"), py::arg("V1"));
}

// Declare assembler function that have multiple scalar types
template <typename T, typename U>
void declare_assembly_functions(py::module& m)
{
  // Coefficient/constant packing
  m.def(
      "pack_coefficients",
      [](const dolfinx::fem::Form<T, U>& form)
      {
        using Key_t = typename std::pair<dolfinx::fem::IntegralType, int>;

        // Pack coefficients
        std::map<Key_t, std::pair<std::vector<T>, int>> coeffs
            = dolfinx::fem::allocate_coefficient_storage(form);
        dolfinx::fem::pack_coefficients(form, coeffs);

        // Move into NumPy data structures
        std::map<Key_t, py::array_t<T, py::array::c_style>> c;
        std::transform(
            coeffs.begin(), coeffs.end(), std::inserter(c, c.end()),
            [](auto& e) -> typename decltype(c)::value_type
            {
              int num_ents = e.second.first.empty()
                                 ? 0
                                 : e.second.first.size() / e.second.second;
              return {e.first, dolfinx_wrappers::as_pyarray(
                                   std::move(e.second.first),
                                   std::array{num_ents, e.second.second})};
            });

        return c;
      },
      py::arg("form"), "Pack coefficients for a Form.");
  m.def(
      "pack_constants",
      [](const dolfinx::fem::Form<T, U>& form) {
        return dolfinx_wrappers::as_pyarray(dolfinx::fem::pack_constants(form));
      },
      py::arg("form"), "Pack constants for a Form.");
  m.def(
      "pack_constants",
      [](const dolfinx::fem::Expression<T, U>& e)
      { return dolfinx_wrappers::as_pyarray(dolfinx::fem::pack_constants(e)); },
      py::arg("e"), "Pack constants for an Expression.");

  // Functional
  m.def(
      "assemble_scalar",
      [](const dolfinx::fem::Form<T, U>& M,
         const py::array_t<T, py::array::c_style>& constants,
         const std::map<std::pair<dolfinx::fem::IntegralType, int>,
                        py::array_t<T, py::array::c_style>>& coefficients)
      {
        return dolfinx::fem::assemble_scalar<T>(
            M, std::span(constants.data(), constants.size()),
            py_to_cpp_coeffs(coefficients));
      },
      py::arg("M"), py::arg("constants"), py::arg("coefficients"),
      "Assemble functional over mesh with provided constants and "
      "coefficients");
  // Vector
  m.def(
      "assemble_vector",
      [](py::array_t<T, py::array::c_style> b,
         const dolfinx::fem::Form<T, U>& L,
         const py::array_t<T, py::array::c_style>& constants,
         const std::map<std::pair<dolfinx::fem::IntegralType, int>,
                        py::array_t<T, py::array::c_style>>& coefficients)
      {
        dolfinx::fem::assemble_vector<T>(
            std::span(b.mutable_data(), b.size()), L,
            std::span(constants.data(), constants.size()),
            py_to_cpp_coeffs(coefficients));
      },
      py::arg("b"), py::arg("L"), py::arg("constants"), py::arg("coeffs"),
      "Assemble linear form into an existing vector with pre-packed constants "
      "and coefficients");
  // MatrixCSR
  m.def(
      "assemble_matrix",
      [](dolfinx::la::MatrixCSR<T>& A, const dolfinx::fem::Form<T, U>& a,
         const py::array_t<T, py::array::c_style>& constants,
         const std::map<std::pair<dolfinx::fem::IntegralType, int>,
                        py::array_t<T, py::array::c_style>>& coefficients,
         const std::vector<
             std::shared_ptr<const dolfinx::fem::DirichletBC<T, U>>>& bcs)
      {
        const std::array<int, 2> data_bs
            = {a.function_spaces().at(0)->dofmap()->index_map_bs(),
               a.function_spaces().at(1)->dofmap()->index_map_bs()};

        if (data_bs[0] != data_bs[1])
          throw std::runtime_error(
              "Non-square blocksize unsupported in Python");

        if (data_bs[0] == 1)
        {
          dolfinx::fem::assemble_matrix(
              A.mat_add_values(), a,
              std::span(constants.data(), constants.size()),
              py_to_cpp_coeffs(coefficients), bcs);
        }
        else if (data_bs[0] == 2)
        {
          auto mat_add = A.template mat_add_values<2, 2>();
          dolfinx::fem::assemble_matrix(
              mat_add, a, std::span(constants.data(), constants.size()),
              py_to_cpp_coeffs(coefficients), bcs);
        }
        else if (data_bs[0] == 3)
        {
          auto mat_add = A.template mat_add_values<3, 3>();
          dolfinx::fem::assemble_matrix(
              mat_add, a, std::span(constants.data(), constants.size()),
              py_to_cpp_coeffs(coefficients), bcs);
        }
        else
          throw std::runtime_error("Block size not supported in Python");
      },
      py::arg("A"), py::arg("a"), py::arg("constants"), py::arg("coeffs"),
      py::arg("bcs"), "Experimental.");
  m.def(
      "insert_diagonal",
      [](dolfinx::la::MatrixCSR<T>& A, const dolfinx::fem::FunctionSpace<U>& V,
         const std::vector<
             std::shared_ptr<const dolfinx::fem::DirichletBC<T, U>>>& bcs,
         T diagonal)
      {
        // NB block size of data ("diagonal") is (1, 1)
        dolfinx::fem::set_diagonal(A.mat_set_values(), V, bcs, diagonal);
      },
      py::arg("A"), py::arg("V"), py::arg("bcs"), py::arg("diagonal"),
      "Experimental.");
  m.def(
      "insert_diagonal",
      [](dolfinx::la::MatrixCSR<T>& A, const py::array_t<std::int32_t>& rows,
         T diagonal)
      {
        dolfinx::fem::set_diagonal(
            A.mat_set_values(), std::span(rows.data(), rows.size()), diagonal);
      },
      py::arg("A"), py::arg("rows"), py::arg("diagonal"), "Experimental.");
  m.def(
      "assemble_matrix",
      [](const std::function<int(const py::array_t<std::int32_t>&,
                                 const py::array_t<std::int32_t>&,
                                 const py::array_t<T>&)>& fin,
         const dolfinx::fem::Form<T, U>& form,
         const std::vector<
             std::shared_ptr<const dolfinx::fem::DirichletBC<T, U>>>& bcs)
      {
        auto f = [&fin](const std::span<const std::int32_t>& rows,
                        const std::span<const std::int32_t>& cols,
                        const std::span<const T>& data)
        {
          return fin(py::array(rows.size(), rows.data()),
                     py::array(cols.size(), cols.data()),
                     py::array(data.size(), data.data()));
        };
        dolfinx::fem::assemble_matrix(f, form, bcs);
      },
      py::arg("fin"), py::arg("form"), py::arg("bcs"),
      "Experimental assembly with Python insertion function. This will be "
      "slow. Use for testing only.");

  // BC modifiers
  m.def(
      "apply_lifting",
      [](py::array_t<T, py::array::c_style> b,
         const std::vector<std::shared_ptr<const dolfinx::fem::Form<T, U>>>& a,
         const std::vector<py::array_t<T, py::array::c_style>>& constants,
         const std::vector<std::map<std::pair<dolfinx::fem::IntegralType, int>,
                                    py::array_t<T, py::array::c_style>>>&
             coeffs,
         const std::vector<std::vector<
             std::shared_ptr<const dolfinx::fem::DirichletBC<T, U>>>>& bcs1,
         const std::vector<py::array_t<T, py::array::c_style>>& x0, T scale)
      {
        std::vector<std::span<const T>> _x0;
        for (auto x : x0)
          _x0.emplace_back(x.data(), x.size());

        std::vector<std::span<const T>> _constants;
        std::transform(constants.begin(), constants.end(),
                       std::back_inserter(_constants),
                       [](auto& c) { return std::span(c.data(), c.size()); });

        std::vector<std::map<std::pair<dolfinx::fem::IntegralType, int>,
                             std::pair<std::span<const T>, int>>>
            _coeffs;
        std::transform(coeffs.begin(), coeffs.end(),
                       std::back_inserter(_coeffs),
                       [](auto& c) { return py_to_cpp_coeffs(c); });

        dolfinx::fem::apply_lifting<T>(std::span(b.mutable_data(), b.size()), a,
                                       _constants, _coeffs, bcs1, _x0, scale);
      },
      py::arg("b"), py::arg("a"), py::arg("constants"), py::arg("coeffs"),
      py::arg("bcs1"), py::arg("x0"), py::arg("scale"),
      "Modify vector for lifted boundary conditions");
  m.def(
      "set_bc",
      [](py::array_t<T, py::array::c_style> b,
         const std::vector<
             std::shared_ptr<const dolfinx::fem::DirichletBC<T, U>>>& bcs,
         const py::array_t<T, py::array::c_style>& x0, T scale)
      {
        if (x0.ndim() == 0)
        {
          dolfinx::fem::set_bc<T>(std::span(b.mutable_data(), b.size()), bcs,
                                  scale);
        }
        else if (x0.ndim() == 1)
        {
          dolfinx::fem::set_bc<T>(std::span(b.mutable_data(), b.size()), bcs,
                                  std::span(x0.data(), x0.shape(0)), scale);
        }
        else
          throw std::runtime_error("Wrong array dimension.");
      },
      py::arg("b"), py::arg("bcs"), py::arg("x0") = py::none(),
      py::arg("scale") = T(1));
}

} // namespace

namespace dolfinx_wrappers
{

void assemble(py::module& m)
{
  // dolfinx::fem::assemble
  declare_assembly_functions<float, float>(m);
  declare_assembly_functions<double, double>(m);
  declare_assembly_functions<std::complex<float>, float>(m);
  declare_assembly_functions<std::complex<double>, double>(m);

  declare_discrete_operators<float, float>(m);
  declare_discrete_operators<double, double>(m);
  declare_discrete_operators<std::complex<float>, float>(m);
  declare_discrete_operators<std::complex<double>, double>(m);
}
} // namespace dolfinx_wrappers
