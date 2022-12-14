// @Copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
//                     All rights reserved.

#pragma once

#include <string>
#include <unordered_map>
using std::string;
using std::unordered_map;

class ArgsParser
{
public:
    ArgsParser() = default;
    ArgsParser(int argc, char **argv);
    int getint(string key);
    string getstring(string key);
    unordered_map<string, string> argmap;
};


