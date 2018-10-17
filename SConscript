#/******************************************************************************
# * SConscript
# *
# * Source of KaHIP -- Karlsruhe High Quality Partitioning.
# *
# ******************************************************************************
# * Copyright (C) 2015 Christian Schulz <christian.schulz@kit.edu>
# *
# * This program is free software: you can redistribute it and/or modify it
# * under the terms of the GNU General Public License as published by the Free
# * Software Foundation, either version 2 of the License, or (at your option)
# * any later version.
# *
# * This program is distributed in the hope that it will be useful, but WITHOUT
# * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
# * more details.
# *
# * You should have received a copy of the GNU General Public License along with
# * this program.  If not, see <http://www.gnu.org/licenses/>.
# *****************************************************************************/


# The main SConscript file for the code.
#
# We simply import the main environment and then define the targets.  This
# submodule contains a sequential matching and contraction code and tests for
# the code.
import platform
import sys

# Get the current platform.
SYSTEM = platform.uname()[0]

Import('env')

# Build a library from the code in lib/.
libkaffpa_files = [ 'lib/data_structure/graph_hierarchy.cpp',
                    'lib/algorithms/strongly_connected_components.cpp',
                    'lib/algorithms/topological_sort.cpp',
                    'lib/algorithms/push_relabel.cpp',
                    'lib/io/graph_io.cpp',
                    'lib/tools/quality_metrics.cpp',
                    'lib/tools/random_functions.cpp',
                    'lib/partition/coarsening/coarsening.cpp',
                    'lib/partition/coarsening/contraction.cpp',
                    'lib/partition/coarsening/edge_rating/edge_ratings.cpp',
                    'lib/partition/coarsening/matching/matching.cpp',
                    'lib/partition/coarsening/matching/random_matching.cpp',
                    'lib/partition/coarsening/matching/gpa/path.cpp',
                    'lib/partition/coarsening/matching/gpa/gpa_matching.cpp',
                    'lib/partition/coarsening/matching/gpa/path_set.cpp',
                    'lib/partition/coarsening/clustering/node_ordering.cpp',
                    'lib/partition/coarsening/clustering/size_constraint_label_propagation.cpp',
                    'lib/partition/uncoarsening/refinement/quotient_graph_refinement/complete_boundary.cpp',
                    'lib/partition/uncoarsening/refinement/quotient_graph_refinement/partial_boundary.cpp',
                    ]

libmlsvm_files = [ 'lib/svm/svm_solver.cpp',
                   'lib/svm/svm_instance.cpp',
                   'lib/svm/svm_summary.cpp',
                   'lib/svm/svm_result.cpp',
                   'lib/svm/svm_convert.cpp',
                   'lib/svm/param_search.cpp',
                   'lib/svm/k_fold.cpp',
                   'lib/svm/k_fold_build.cpp',
                   'lib/svm/k_fold_import.cpp',
                   'lib/io/svm_io.cpp',
                   'lib/svm/results.cpp',
                   'lib/svm/svm_flann.cpp',
                   'lib/svm/svm_refinement.cpp',
                   'extern/libsvm-3.22/src/svm.cpp' ]

prepare_files = [  'lib/svm/svm_flann.cpp' ]

if env['program'] == 'mlsvm':
        env.Append(CXXFLAGS = '-DMODE_MLSVM')
        env.Append(CCFLAGS  = '-DMODE_MLSVM')
        env.Program('mlsvm', ['app/mlsvm.cpp']+libkaffpa_files+libmlsvm_files, LIBS=['libargtable2','gomp'])

if env['program'] == 'single_level':
        env.Append(CXXFLAGS = '-DMODE_MLSVM')
        env.Append(CCFLAGS  = '-DMODE_MLSVM')
        env.Program('single_level_svm', ['app/single_level_svm.cpp']+libkaffpa_files+libmlsvm_files, LIBS=['libargtable2','gomp','pthread'])

if env['program'] == 'prepare':
        env.Append(CXXFLAGS = '-DMODE_MLSVM')
        env.Append(CCFLAGS  = '-DMODE_MLSVM')
        env.Program('prepare', ['app/prepare.cpp']+prepare_files, LIBS=['libargtable2','gomp'])
