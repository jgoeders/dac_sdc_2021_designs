## Compiling command for image loading and bbox computation C lib
## It may require the root account to compile this C code
g++ -shared -O3 load_image_pingpong3.cpp -o load_image_pingpong3.so -fPIC `pkg-config opencv --cflags --libs`-lpthread -fpermissive

 
