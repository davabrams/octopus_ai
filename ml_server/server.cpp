// C++ tensorflow server
// build with:
//          g++ -std=c++14 server.cpp -oserver

#include "server_internal.cpp"


int main()
{
    server_loop();
    return 0;
}