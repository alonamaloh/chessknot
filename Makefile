CXX = g++
CXXFLAGS = -std=c++20 -O2 -Wall -Wextra -march=native

CORE_SRC = core/board.cpp core/movegen.cpp
CORE_OBJ = $(CORE_SRC:.cpp=.o)

SEARCH_SRC = search/tt.cpp search/search.cpp
SEARCH_OBJ = $(SEARCH_SRC:.cpp=.o)

all: perft test_search

perft: perft.o $(CORE_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

test_search: test_search.o $(CORE_OBJ) $(SEARCH_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f perft test_search *.o core/*.o search/*.o

.PHONY: all clean
