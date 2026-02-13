CXX = g++
CXXFLAGS = -std=c++20 -O2 -Wall -Wextra -march=native

HDF5_CFLAGS = -I/usr/include/hdf5/serial
HDF5_LIBS = -L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5_cpp -lhdf5

CORE_SRC = core/board.cpp core/movegen.cpp
CORE_OBJ = $(CORE_SRC:.cpp=.o)

NN_SRC = nn/mlp.cpp
NN_OBJ = $(NN_SRC:.cpp=.o)

SEARCH_SRC = search/tt.cpp search/search.cpp
SEARCH_OBJ = $(SEARCH_SRC:.cpp=.o)

all: perft test_search selfplay

perft: perft.o $(CORE_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

test_search: test_search.o $(CORE_OBJ) $(SEARCH_OBJ) $(NN_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

selfplay: selfplay.o $(CORE_OBJ) $(SEARCH_OBJ) $(NN_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(HDF5_LIBS)

selfplay.o: selfplay.cpp
	$(CXX) $(CXXFLAGS) $(HDF5_CFLAGS) -c -o $@ $<

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f perft test_search selfplay *.o core/*.o search/*.o nn/*.o

.PHONY: all clean
