#ifndef _PATHKIT_CPP_
#define _PATHKIT_CPP_

#define MAX(a,b)  ((a) < (b) ? (b) : (a))

#include <iostream>
#include <algorithm>

namespace shim {

void split_ext(std::string path, std::string &name, std::string &ext) {
    size_t d = path.find('.');
    name = path.substr(0, d);
    ext = path.substr(d);
}

void replace_string(std::string &path, char src, char dst) {
    size_t p;
    while(true) {
        p = path.find(src);
        if(p < 0) break;
        
        path[p] = dst;
    }
}

std::string dirname(std::string path) {
    size_t P = std::max(std::max(path.rfind('/'), path.rfind('\\')), (size_t)0);
    return path.substr(0, P); 
}

std::string basename(std::string path) {
    size_t P = std::max(path.rfind('/'), path.rfind('\\'));
    if(P == -1) return path;
    return path.substr(P+1, path.length()-P-1);
}

}

#endif
