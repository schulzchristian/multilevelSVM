/******************************************************************************
 * graph_io.cpp 
 *
 * Source of KaHIP -- Karlsruhe High Quality Partitioning.
 *
 ******************************************************************************
 * Copyright (C) 2013-2015 Christian Schulz <christian.schulz@kit.edu>
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation, either version 2 of the License, or (at your option)
 * any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program.  If not, see <http://www.gnu.org/licenses/>.
 *****************************************************************************/

#include <sstream>
#include <unordered_set>
#include "graph_io.h"

graph_io::graph_io() {
                
}

graph_io::~graph_io() {
                
}

int graph_io::writeGraphWeighted(graph_access & G, std::string filename) {
        std::ofstream f(filename.c_str());
        f << G.number_of_nodes() <<  " " <<  G.number_of_edges()/2 <<  " 11" <<  std::endl;

        forall_nodes(G, node) {
                f <<  G.getNodeWeight(node) ;
                forall_out_edges(G, e, node) {
                        f << " " <<   (G.getEdgeTarget(e)+1) <<  " " <<  G.getEdgeWeight(e) ;
                } endfor 
                f <<  std::endl;
        } endfor

        f.close();
        return 0;
}

int graph_io::writeGraph(graph_access & G, std::string filename) {
        std::ofstream f(filename.c_str());
        f << G.number_of_nodes() <<  " " <<  G.number_of_edges()/2 << std::endl;

        forall_nodes(G, node) {
                forall_out_edges(G, e, node) {
                        f <<   (G.getEdgeTarget(e)+1) << " " ;
                } endfor 
                f <<  std::endl;
        } endfor

        f.close();
        return 0;
}

int graph_io::writeGraphGDF(const graph_access & G_min, const graph_access & G_maj, std::string filename) {
        std::ofstream f(filename.c_str());

        size_t min_nodes = G_min.number_of_nodes();

	f << "nodedef>name VARCHAR,class VARCHAR,partition VARCHAR, weight DOUBLE";
	for (size_t i = 0; i < G_min.getFeatureVec(0).size(); ++i) {
		f << ",feature" << i << " DOUBLE";
	}
	f << std::endl;

	// NODES
	forall_nodes(G_min, node) {
		f <<  node << ",-1," << G_min.getPartitionIndex(node) << "," << G_min.getNodeWeight(node);
		for (auto &feature : G_min.getFeatureVec(node)) {
			f << "," << feature;
		}
		f << std::endl;
	} endfor

	forall_nodes(G_maj, node) {
		f <<  node + min_nodes << ",1," << G_maj.getPartitionIndex(node) << "," << G_maj.getNodeWeight(node);
		for (auto &feature : G_maj.getFeatureVec(node)) {
			f << "," << feature;
		}
		f << std::endl;
	} endfor


	f << "edgedef>from VARCHAR,to VARCHAR" << std::endl;

	// EDGES
	forall_nodes(G_min, node) {
                forall_out_edges(G_min, e, node) {
                        f << node << "," << G_min.getEdgeTarget(e) << std::endl;
                } endfor 
        } endfor
	forall_nodes(G_maj, node) {
                forall_out_edges(G_maj, e, node) {
                        f << node + min_nodes << "," << G_maj.getEdgeTarget(e) + min_nodes << std::endl;
                } endfor 
        } endfor
        f.close();
        return 0;
}


int graph_io::readPartition(graph_access & G, std::string filename) {
        std::string line;

        // open file for reading
        std::ifstream in(filename.c_str());
        if (!in) {
                std::cerr << "Error opening file " << filename << std::endl;
                return 1;
        }

        PartitionID max = 0;
        forall_nodes(G, node) {
                // fetch current line
                std::getline(in, line);
                if (line[0] == '%') { //Comment
                        node--;
                        continue;
                }

                // in this line we find the block of Node node 
                G.setPartitionIndex(node, (PartitionID) atol(line.c_str()));

                if(G.getPartitionIndex(node) > max)
                        max = G.getPartitionIndex(node);
        } endfor

        G.set_partition_count(max+1);
        in.close();

        return 0;
}

int graph_io::readGraphWeighted(graph_access & G, std::string filename) {
        std::string line;

        // open file for reading
        std::ifstream in(filename.c_str());
        if (!in) {
                std::cerr << "Error opening " << filename << std::endl;
                return 1;
        }

        long nmbNodes;
        long nmbEdges;

        std::getline(in,line);
        //skip comments
        while( line[0] == '%' ) {
                std::getline(in, line);
        }

        int ew = 0;
        std::stringstream ss(line);
        ss >> nmbNodes;
        ss >> nmbEdges;
        ss >> ew;

        nmbEdges *= 2; //since we have forward and backward edges
        if( nmbEdges > std::numeric_limits<int>::max() || nmbNodes > std::numeric_limits<int>::max()) {
                std::cerr <<  "The graph is too large. Currently only 32bit supported!"  << std::endl;
                exit(0);
        }

        bool read_ew = false;
        bool read_nw = false;

        if(ew == 1) {
                read_ew = true;
        } else if (ew == 11) {
                read_ew = true;
                read_nw = true;
        } else if (ew == 10) {
                read_nw = true;
        }
        
        NodeID node_counter   = 0;
        EdgeID edge_counter   = 0;
        long long total_nodeweight = 0;

        G.start_construction(nmbNodes, nmbEdges);

        while(  std::getline(in, line)) {
       
                if (line[0] == '%') { // a comment in the file
                        continue;
                }

                NodeID node = G.new_node(); node_counter++;
                G.setPartitionIndex(node, 0);

                std::stringstream ss(line);

                NodeWeight weight = 1;
                if( read_nw ) {
                        ss >> weight;
                        total_nodeweight += weight;
                        if( total_nodeweight > (long long) std::numeric_limits<NodeWeight>::max()) {
                                std::cerr <<  "The sum of the node weights is too large (it exceeds the node weight type)."  << std::endl;
                                std::cerr <<  "Currently not supported. Please scale your node weights."  << std::endl;
                                exit(0);
                        }
                }
                G.setNodeWeight(node, weight);

                NodeID target;
                while( ss >> target ) {
                        //check for self-loops
                        if(target-1 == node) {
                                std::cerr <<  "The graph file contains self-loops. This is not supported. Please remove them from the file."  << std::endl;
                        }

                        EdgeWeight edge_weight = 1;
                        if( read_ew ) {
                                ss >> edge_weight;
                        }
                        edge_counter++;
                        EdgeID e = G.new_edge(node, target-1);

                        G.setEdgeWeight(e, edge_weight);
                }

                if(in.eof()) {
                        break;
                }
        }

        /*
        if( edge_counter != (EdgeID) nmbEdges ) {
                std::cerr <<  "number of specified edges mismatch"  << std::endl;
                std::cerr <<  edge_counter <<  " " <<  nmbEdges  << std::endl;
                exit(0);
        }
        */

        if( node_counter != (NodeID) nmbNodes) {
                std::cerr <<  "number of specified nodes mismatch"  << std::endl;
                std::cerr <<  node_counter <<  " " <<  nmbNodes  << std::endl;
                exit(0);
        }


        G.finish_construction();
        return 0;
}


void graph_io::writePartition(graph_access & G, std::string filename) {
        std::ofstream f(filename.c_str());
        std::cout << "writing partition to " << filename << " ... " << std::endl;

        forall_nodes(G, node) {
                f << G.getPartitionIndex(node) <<  std::endl;
        } endfor

        f.close();
}


int graph_io::readFeatures(graph_access & G, const std::string & filename) {
        std::string line;

        // open file for reading
        std::ifstream in(filename);
        if (!in) {
                std::cerr << "Error opening file " << filename << std::endl;
                return 1;
        }

        std::getline(in, line);
        std::stringstream s(line);
        int nodes = 0;
        int features = 0;
        s >> nodes;
        s >> features;

        forall_nodes(G, node) {
                // fetch current line
                std::getline(in, line);
                std::stringstream ss(line);

                FeatureVec vec(features);
                for (int i = 0; i < features; i++) {
                        ss >> vec[i];
                }

                G.setFeatureVec(node, vec);
        } endfor
}

int graph_io::readFeatures(graph_access & G, const std::vector<FeatureVec> & data) {
        forall_nodes(G, node) {
                G.setFeatureVec(node, data[node]);
        } endfor
        return 0;
}

int graph_io::readGraphFromVec(graph_access & G, const std::vector<std::vector<Edge>> & data, EdgeID num_edges) {
        G.start_construction(data.size(), num_edges);

        EdgeID last = 0;
        for (auto& nodeData : data) {
                NodeID node = G.new_node();
                G.setPartitionIndex(node, 0);
                G.setNodeWeight(node, 1);

                for (auto & edge : nodeData) {
                        EdgeID e = G.new_edge(node, edge.target);
                        G.setEdgeWeight(e, edge.weight);
			// std::cout << edge.target << "," << edge.weight <<std::endl;
                        last = e;
                }
        }

        G.finish_construction();
        return 0;
}

EdgeID graph_io::makeEdgesBidirectional(std::vector<std::vector<Edge>> & data) {
        size_t pre = 0;
        size_t post = 0;

        std::vector<std::unordered_set<NodeID>> neighbors(data.size());

        for (NodeID from = 0; from < data.size(); ++from) {
                for (EdgeID edge = 0; edge < data[from].size(); ++edge) {
                        NodeID target = data[from][edge].target;
                        neighbors[from].insert(target);
                        pre++;
                }
        }


        for (NodeID from = 0; from < data.size(); ++from) {
                for (EdgeID edge = 0; edge < data[from].size(); ++edge) {
                        Edge e = data[from][edge];
                        if (neighbors[e.target].find(from) == neighbors[from].end()) {
                                Edge back;
                                back.target = from;
                                back.weight = e.weight;
                                data[e.target].push_back(back);
                                neighbors[e.target].insert(from);
                        }
                }
        }

        for (auto& nodeData : data) {
                for (auto & edge : nodeData) {
                        post++;
                }
        }

        std::cout << "edges pre: " << pre << " post: " << post << std::endl;
        return post;
}
