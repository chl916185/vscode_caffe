cmake_minimum_required(VERSION 2.8.3)
project(classify)

#设置caffe的根目录，需改成你的路径
set(Caffe_DIR /home/chl/workspaces/caffe)
#设置OpenCV的根目录，需改成你的路径
#set(OpenCV_DIR /home/chl/workspaces/soft/opencv-3.1.0/build)

include_directories ( 
    ${Caffe_DIR}/include
    /usr/local/include  
    ${Caffe_DIR}/build/src
    /usr/local/cuda/include 
)  
link_directories(/home/chl/workspaces/caffe/build/lib)
find_package(OpenCV REQUIRED)
find_package(HDF5 COMPONENTS HL REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${HDF5_INCLUDE_DIRS} ${HDF5_HL_INCLUDE_DIR})
#list(APPEND Caffe_LINKER_LIBS ${HDF5_LIBRARIES})
#list(APPEND Caffe_LINKER_LIBS ${HDF5_HL_LIBRARIES}) #added

include_directories(include 
    ${OpenCV_INCLUDE_DIRS}
)

add_executable(classify main.cpp)
target_link_libraries(classify
    ${OpenCV_LIBS}
)

# change the directory of libcaffe.so in your computer
target_link_libraries(classify ${Caffe_DIR}/build/lib/libcaffe.so   
    /usr/lib/x86_64-linux-gnu/libglog.so  
    /usr/lib/x86_64-linux-gnu/libboost_system.so  
    /usr/lib/x86_64-linux-gnu/libtiff.so
    )
    
target_link_libraries(classify
    ${HDF5_HL_LIBRARIES}
    ${HDF5_HL_LIBRARIES}
) 