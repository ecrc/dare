// @Copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
//                     All rights reserved.

#include "darecommon.h"
#include <iostream>
using std::cout;
using std::endl;


ArgsParser::ArgsParser(int argc, char **argv)
{
    for (int i = 1; i < argc; i++)
    {
        string tmp = string(argv[i]);
        if (tmp.substr(0, 2) != "--")
            continue;
        else
        {
            int s = 0;
            while (s < tmp.size() && tmp[s] != '=')
                s++;
            if (s == tmp.size())
                continue;
            argmap[tmp.substr(2, s - 2)] = tmp.substr(s + 1, tmp.size() - 2 - 1);
        }
    }
}
int ArgsParser::getint(string key)
{
    if (argmap.find(key) == argmap.end())
    {
        cout << "input key error, key: " << key << endl;
        exit(0);
    }
    return atoi(argmap[key].c_str());
}
string ArgsParser::getstring(string key)
{
    if (argmap.find(key) == argmap.end())
    {
        cout << "input key error, key:" << key << endl;
        exit(0);
    }
    return argmap[key];
}

