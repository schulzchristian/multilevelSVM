#include "catch.hpp"
#include "svm/svm_convert.h"

TEST_CASE( "svm_convert converts between FeatureVec and svm_feature", "[multi-file:2]" ) {
	FeatureVec vec{0,0,0,0,1,1.5};
	svm_feature dat(3);
	dat[0].index = 5;
	dat[0].value = 1;
	dat[1].index = 6;
	dat[1].value = 1.5;
	dat[2].index = -1;
	dat[2].value = 0;

	SECTION( "convert from FeatureVec to svm_feature " ) {
		svm_feature computed = svm_convert::feature_to_node(vec);
		// REQUIRE( computed == dat );
		REQUIRE( computed.size() == 3 );
		REQUIRE( computed[0].index == 5 );
		REQUIRE( computed[1].value == 1.5 );
	}

	SECTION( "convert from svm_feature to FeatureVec " ) {
		FeatureVec computed = svm_convert::node_to_feature(dat);
		// REQUIRE( computed == vec );
		REQUIRE( computed.size() == 6);
		REQUIRE( computed[0] == 0);
		REQUIRE( computed[1] == 0);
		REQUIRE( computed[5] == 1.5);
	}
}
