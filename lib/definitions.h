/******************************************************************************
 * definitions.h
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

#ifndef DEFINITIONS_H_CHR
#define DEFINITIONS_H_CHR

#include <limits>
#include <queue>
#include <vector>

#include "limits.h"
#include "tools/macros_assertions.h"
#include "stdio.h"

// allows us to disable most of the output during partitioning
#ifdef KAFFPAOUTPUT
        #define PRINT(x) x
#else
        #define PRINT(x) do {} while (false);
#endif

/**********************************************
 * Constants
 * ********************************************/
//Types needed for the graph ds
typedef unsigned int  NodeID;
typedef double    EdgeRatingType;
typedef unsigned int  EdgeID;
typedef unsigned int  PathID;
typedef unsigned int  PartitionID;
typedef unsigned int  NodeWeight;
typedef double     EdgeWeight;
typedef double FeatureData;
typedef std::vector<FeatureData> FeatureVec;
typedef EdgeWeight  Gain;
typedef int     Color;
typedef unsigned int  Count;
typedef std::vector<NodeID> boundary_starting_nodes;
typedef long FlowType;

const EdgeID UNDEFINED_EDGE            = std::numeric_limits<EdgeID>::max();
const NodeID NOTMAPPED                 = std::numeric_limits<EdgeID>::max();
const NodeID UNDEFINED_NODE            = std::numeric_limits<NodeID>::max();
const NodeID UNASSIGNED                = std::numeric_limits<NodeID>::max();
const NodeID ASSIGNED                  = std::numeric_limits<NodeID>::max()-1;
const PartitionID INVALID_PARTITION    = std::numeric_limits<PartitionID>::max();
const PartitionID BOUNDARY_STRIPE_NODE = std::numeric_limits<PartitionID>::max();
const int NOTINQUEUE           = std::numeric_limits<int>::max();
const int ROOT             = 0;

// for graph_access
struct Node {
        EdgeID firstEdge;
        NodeWeight weight;
};

struct Edge {
        NodeID target;
        EdgeWeight weight;
};

//for the gpa algorithm
struct edge_source_pair {
        EdgeID e;
        NodeID source;
};

struct source_target_pair {
        NodeID source;
        NodeID target;
};

//matching array has size (no_of_nodes), so for entry in this table we get the matched neighbor
typedef std::vector<NodeID> CoarseMapping;
typedef std::vector<NodeID> Matching;
typedef std::vector<NodeID> NodePermutationMap;

typedef double ImbalanceType;

//Coarsening
typedef enum {
        EXPANSIONSTAR,
        EXPANSIONSTAR2,
        WEIGHT,
        REALWEIGHT,
        PSEUDOGEOM,
        EXPANSIONSTAR2ALGDIST,
        SEPARATOR_MULTX,
        SEPARATOR_ADDX,
        SEPARATOR_MAX,
        SEPARATOR_LOG,
        SEPARATOR_R1,
        SEPARATOR_R2,
        SEPARATOR_R3,
        SEPARATOR_R4,
        SEPARATOR_R5,
        SEPARATOR_R6,
        SEPARATOR_R7,
        SEPARATOR_R8
} EdgeRating;

typedef enum {
        PERMUTATION_QUALITY_NONE,
        PERMUTATION_QUALITY_FAST,
        PERMUTATION_QUALITY_GOOD
} PermutationQuality;

typedef enum {
        MATCHING_RANDOM,
        MATCHING_GPA,
        MATCHING_RANDOM_GPA,
        LP_CLUSTERING,
        SIMPLE_CLUSTERING,
        LOW_DIAMETER
} MatchingType;

typedef enum {
        STOP_RULE_SIMPLE_FIXED
} StopRule;

typedef enum {
        RANDOM_NODEORDERING,
        DEGREE_NODEORDERING
} NodeOrderingType;

typedef enum {
	KFOLD,
	KFOLD_IMPORT,
	TRAIN_TEST_SPLIT,
	ONCE
} ValidationType;

typedef enum {
	UD,
	BAYES,
	FIX
} RefinementType;

#endif
