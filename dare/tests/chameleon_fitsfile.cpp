// @Copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
//                     All rights reserved.

/**
 * This file is used to test chameleon fits read/write.
 * */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <vector>
#include <unistd.h>
#include <sys/resource.h>
#include <coreblas/lapacke.h>
#include <chameleon.h>
#include <chameleon/timer.h>
#include <chameleon/flops.h>
#include <iostream>

#include "common/darecommon.h"
#include "chameleonsrc/chameio.hpp"
#include "chameleonsrc/testings.h"


using std::cout;
using std::endl;
using std::vector;
using std::to_string;

/**
 * A numpy view of data is 
 * shape: 2 x 5
 * [[0 1 2 3 4]
 * [5 6 7 8 9]]
 * datatype: int64
 * 
 * fits read dimension is arranged in matrix's transpose order.
 * so we should save A.T in python as fits file.
 * Then we can read correct value as colmajor.
 * 2,5
 * elements 0 , 0
 * elements 1 , 5
 * elements 2 , 1
 * elements 3 , 6
 * elements 4 , 2
 * elements 5 , 7
 * elements 6 , 3
 * elements 7 , 8
 * elements 8 , 4
 * elements 9 , 9
 * */
int main(int argc, char** argv){
    FitsFile f;
    ArgsParser argparser(argc, argv);
    std::string filename = argparser.getstring("filename");
    f.set(filename, "r");
    std::vector<long> shape_f = f.get_data_shape(0);
    cout << "fits read dimension is arranged in matrix's transpose order." << endl;
    cout << "if matrix is m by n in python, then it should be n, m here." << endl;
    cout << shape_f[0] << " , "<< shape_f[1]<< endl;
    auto m = shape_f[0]; auto n = shape_f[1];
    // read fits
    std::vector<long> v = f.get_data<long>("PRIMARY");
    for(long i=0; i<(v.size() > 10 ? 10 : v.size()); i++){
        cout << "elements " << i << " , " << v[i] << endl;
    }
    // write a new fits
    ships::io::FitsFile fw("newmat.fits","w+");
    fw.set_data({m,n},v.data(),"PRIMARY");
    return 0;
}