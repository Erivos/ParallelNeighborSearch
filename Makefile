CXX=mpic++
CXXFLAGS=-larmadillo -lmlpack -fopenmp

SRCSDIR=src
DESTDIR=target

SRCS=$(SRCSDIR)/ParallelNeighborSearch.cpp


parallelSearch:
	@mkdir -p $(DESTDIR)
	$(CXX) -g $(SRCS) $(CXXFLAGS) -o $(DESTDIR)/$@

clean:
	rm -rf $(DESTDIR)/parallelSearch

cleanall:
	rm -rf $(DESTDIR)