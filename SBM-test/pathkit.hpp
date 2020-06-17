#ifndef _PATHKIT_HPP_
#define _PATHKIT_HPP_

#define MAX(a,b)  ((a) < (b) ? (b) : (a))

#include <iostream>
#include <algorithm>

namespace shim {

void split_ext(std::string path, std::string &name, std::string &ext);
void replace_string(std::string &path, char src, char dst);
std::string dirname(std::string path);
std::string basename(std::string path);

}

#endif
