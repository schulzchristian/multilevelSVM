/******************************************************************************
 * partition_config.h
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

#ifndef PARTITION_CONFIG_DI1ES4T0
#define PARTITION_CONFIG_DI1ES4T0

#include "definitions.h"

// Configuration for the partitioning.
struct PartitionConfig
{
        PartitionConfig() {}


        //============================================================
        //=======================MATCHING=============================
        //============================================================
        bool edge_rating_tiebreaking = false;

        EdgeRating edge_rating = WEIGHT;

        PermutationQuality permutation_quality = PERMUTATION_QUALITY_FAST;

        MatchingType matching_type = CLUSTER_COARSENING;

        bool first_level_random_matching = false;

        bool rate_first_level_inner_outer = false;

        NodeWeight max_vertex_weight;

        NodeWeight largest_graph_weight;

        NodeWeight work_load;

        unsigned aggressive_random_levels = 3;

        bool disable_max_vertex_weight_constraint = false;

        //=======================================
        //=============STOP RULES================
        //=======================================

        StopRule stop_rule = STOP_RULE_SIMPLE;

        int num_vert_stop_factor = 20;

        int fix_num_vert_stop = 500;

        bool no_change_convergence = false;

        NodeWeight upper_bound_partition = std::numeric_limits<NodeWeight>::max()/2;

        //=======================================
        //============PAR_PSEUDOMH / MH =========
        //=======================================

        bool combine = false; // in this case the second index is filled and edges between both partitions are not contracted
        //=======================================
        //===============MISC====================
        //=======================================

        int num_experiments = 1;

        int kfold_iterations = 5;

        std::string input_partition;

        // number of blocks the graph should be partitioned in
        PartitionID k = 1;

        int seed = 0;

        std::string graph_filename;

        std::string filename_output = "";

        double time_limit;

        int timeout = 0;

        //=======================================
        //===========SNW PARTITIONING============
        //=======================================

        NodeOrderingType node_ordering = DEGREE_NODEORDERING;

        int cluster_coarsening_factor = 18;

        bool ensemble_clusterings = false;

        int label_iterations = 10;

        int number_of_clusterings = 1;

        double balance_factor = 0;

        int repetitions = 1;

        //=======================================
        //==========LABEL PROPAGATION============
        //=======================================

        NodeWeight cluster_upperbound = std::numeric_limits<NodeWeight>::max()/2;

        //=======================================
        //============LOW DIAMETER===============
        //=======================================

        double diameter_upperbound = 20;

        //=======================================
        //=======================================
        //=======================================

        bool graph_allready_partitioned = false;

        bool balance_edges;

        bool gpa_grow_paths_between_blocks = true;

        bool bidirectional = false;

        //=======================================
        //===============MLSVM===================
        //=======================================

        float sample_percent = 0.1f;

        int num_nn = 10;

        bool import_kfold = false;

        int num_skip_ms = 10000;

        bool inherit_ud = true;


        void LogDump(FILE *out) const {
        }
};


#endif /* end of include guard: PARTITION_CONFIG_DI1ES4T0 */
