// Copyright (C) 2013 Chris N Richardson
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
//
// First added:  2012-06-01
// Last changed: 2013-10-23

#ifdef HAS_HDF5

#include<boost/lexical_cast.hpp>

#include<dolfin/common/Array.h>

#include "HDF5Attribute.h"
#include "HDF5Interface.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
template <typename T>
void HDF5Attribute::set_value(const std::string attribute_name, 
                              const T& attribute_value)
{
  if(!HDF5Interface::has_dataset(hdf5_file_id, dataset_name))
  {
    dolfin_error("HDF5File.cpp", 
                 "set attribute on dataset",
                 "Dataset does not exist");
  }
  
  if(HDF5Interface::has_attribute(hdf5_file_id, dataset_name, 
                                  attribute_name))
  {
    HDF5Interface::delete_attribute(hdf5_file_id, dataset_name, 
                                    attribute_name);
  }
  
  HDF5Interface::add_attribute(hdf5_file_id, dataset_name, 
                               attribute_name, attribute_value);
}
//-----------------------------------------------------------------------------
template <typename T>
void HDF5Attribute::get_value(const std::string attribute_name, 
                              T& attribute_value) const
{
  HDF5Interface::get_attribute(hdf5_file_id, dataset_name, attribute_name,
                               attribute_value);
}
//-----------------------------------------------------------------------------
void HDF5Attribute::set(const std::string attribute_name, 
                        const double value)
{
  set_value(attribute_name, value);
}
//-----------------------------------------------------------------------------
void HDF5Attribute::set(const std::string attribute_name, 
                        const Array<double>& value)
{
  std::vector<double> value_vec(value.data(), value.data() + value.size());
  set_value(attribute_name, value_vec);
}
//-----------------------------------------------------------------------------
void HDF5Attribute::set(const std::string attribute_name, 
                        const std::string value)
{
  set_value(attribute_name, value);
}
//-----------------------------------------------------------------------------
void HDF5Attribute::get(const std::string attribute_name, double& value) const
{
  get_value(attribute_name, value);
}
//-----------------------------------------------------------------------------
void HDF5Attribute::get(const std::string attribute_name, std::vector<double>& value) const
{
  get_value(attribute_name, value);
}
//-----------------------------------------------------------------------------
void HDF5Attribute::get(const std::string attribute_name, std::string& value) const
{
  const std::string attribute_type = type(attribute_name);
  if(attribute_type == "string")
    get_value(attribute_name, value);
  else if(attribute_type == "float")
  {
    double float_value;
    get_value(attribute_name, float_value);
    value = boost::lexical_cast<std::string>(float_value);
  }
  else if(attribute_type == "vector")
  {
    std::vector<double> vector_value;
    get_value(attribute_name, vector_value);
    value = "";
    const unsigned int nlast = vector_value.size() - 1;
    for(unsigned int i = 0; i < nlast; ++i)
    {
      value += boost::lexical_cast<std::string>(vector_value[i]) + ", ";
    }
    value += boost::lexical_cast<std::string>(vector_value[nlast]);
  }
  else
  {
    value = "Unsupported";
  }
  
}
//-----------------------------------------------------------------------------
const std::string HDF5Attribute::str(const std::string attribute_name) const
{
  std::string str_result;
  get(attribute_name, str_result);
  return str_result;
}
//-----------------------------------------------------------------------------
const std::string HDF5Attribute::type(const std::string attribute_name) const
{
  return HDF5Interface::get_attribute_type(hdf5_file_id, dataset_name, attribute_name);
}
//-----------------------------------------------------------------------------

#endif
