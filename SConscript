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
from os import listdir
from os.path import isfile, join

# Get the current platform.
SYSTEM = platform.uname()[0]

Import('env')

# Build a library from the code in lib/.
libkaffpa_files = [ 'lib/data_structure/graph_hierarchy.cpp',
                    'lib/algorithms/strongly_connected_components.cpp',
                    'lib/algorithms/topological_sort.cpp',
                    'lib/algorithms/push_relabel.cpp',
                    'lib/algorithms/jarnik_prim.cpp',
                    'lib/io/graph_io.cpp',
                    'lib/tools/quality_metrics.cpp',
                    'lib/tools/random_functions.cpp',
                    'lib/partition/partition_config.cpp',
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
                    'lib/partition/coarsening/clustering/simple_clustering.cpp',
                    'lib/partition/coarsening/clustering/low_diameter_clustering.cpp',
                    'lib/partition/uncoarsening/refinement/quotient_graph_refinement/complete_boundary.cpp',
                    'lib/partition/uncoarsening/refinement/quotient_graph_refinement/partial_boundary.cpp',
                    ]

base_files = Split("""
                   lib/svm/k_fold.cpp
                   lib/svm/k_fold_build.cpp
                   lib/svm/k_fold_import.cpp
                   lib/svm/k_fold_once.cpp
                   lib/svm/svm_flann.cpp
                   lib/io/svm_io.cpp
                   lib/svm/svm_convert.cpp
                   lib/svm/results.cpp
""")

libkasvm_files = base_files + [
	       	   'lib/svm/svm_solver.cpp',
                   'lib/svm/svm_solver_libsvm.cpp',
                   'lib/svm/svm_solver_thunder.cpp',
                   'lib/svm/svm_instance.cpp',
                   'lib/svm/svm_summary.cpp',
                   'lib/svm/svm_result.cpp',
                   'lib/svm/param_search.cpp',
                   'lib/svm/svm_refinement.cpp',
                   'lib/svm/ud_refinement.cpp',
                   'lib/svm/bayes_refinement.cpp',
                   'lib/svm/fix_refinement.cpp',
                   'extern/libsvm-3.22/src/svm.cpp' ]

prepare_files = [  'lib/svm/svm_flann.cpp' ]

# test_files = [join('test',f) for f in listdir('../test/') if f.endswith(".cpp")]
test_files = ['test/svm_convert_test.cpp',
              'test/contraction_test.cpp' ]

if env['program'] == 'kasvm':
        env.Library('kasvm', libkaffpa_files+libkasvm_files, LIBS=['libargtable2','thundersvm','bayesopt','nlopt','gomp'])

        env_prog = env.Clone()
        env_prog.Append(CXXFLAGS = '-DMODE_KASVM')
        env_prog.Append(CCFLAGS  = '-DMODE_KASVM')
        env_prog.Append(CXXFLAGS = '-DSVM_SOLVER=svm_solver_thunder')
        env_prog.Append(CXXFLAGS = '-DSVM_MODEL=SVC')
        env_prog.Append(LIBPATH=['.'])
        env_prog.Program('kasvm', ['app/kasvm.cpp'], LIBS=['kasvm', 'libargtable2','thundersvm','bayesopt','nlopt','gomp'])

if env['program'] == 'single_level':
        env.Library('kasvm', libkaffpa_files+libkasvm_files, LIBS=['libargtable2','thundersvm','bayesopt','nlopt','gomp'])

        env_prog = env.Clone()
        env_prog.Append(CXXFLAGS = '-DMODE_KASVM')
        env_prog.Append(CCFLAGS  = '-DMODE_KASVM')
        env_prog.Append(CXXFLAGS = '-DSVM_SOLVER=svm_solver_thunder')
        env_prog.Append(CXXFLAGS = '-DSVM_MODEL=SVC')
        env_prog.Append(LIBPATH=['.'])
        env_prog.Program('single_level_svm', ['app/single_level_svm.cpp'], LIBS=['kasvm', 'libargtable2','thundersvm','bayesopt','nlopt','gomp','pthread'])

if env['program'] == 'prepare':
        env.Program('prepare', ['app/prepare.cpp']+prepare_files, LIBS=['libargtable2','gomp'])

if env['program'] == 'test':
        env.Library('kasvm', libkaffpa_files+libkasvm_files, LIBS=['libargtable2','thundersvm','bayesopt','nlopt','gomp'])

        env.Program('tests', ['test/test.cpp']+test_files+libkaffpa_files+libkasvm_files, LIBS=['libargtable2','thundersvm','bayesopt','nlopt','gomp'])
