# ======================================================================== #
# Copyright 2019 Ingo Wald                                                 #
#                                                                          #
# Licensed under the Apache License, Version 2.0 (the "License");          #
# you may not use this file except in compliance with the License.         #
# You may obtain a copy of the License at                                  #
#                                                                          #
#     http://www.apache.org/licenses/LICENSE-2.0                           #
#                                                                          #
# Unless required by applicable law or agreed to in writing, software      #
# distributed under the License is distributed on an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
# See the License for the specific language governing permissions and      #
# limitations under the License.                                           #
# ======================================================================== #

include_directories(${PROJECT_SOURCE_DIR}/owl)

cuda_compile_and_embed(ptxCode
  deviceCode.cu
  )

add_executable(sample01-trueknn
  hostCode.cpp
  ${ptxCode}
  )

target_link_libraries(sample01-trueknn
  ${OWL_LIBRARIES}
  )

add_test(sample01-trueknn
  ${CMAKE_BINARY_DIR}/sample01-trueknn)



