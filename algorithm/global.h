#ifndef GLOBAL_H_
#define GLOBAL_H_

#include <boost/property_tree/ptree.hpp>

namespace pt = boost::property_tree;

namespace MavesS {

/**
 * \brief Namespace which contains some configuration variables that are set in
 * wave_semilinear.cc.
 */
namespace global {

/** \brief variable for the config file */
extern pt::ptree config;

} /* global */
} /* MavesS */

#endif /* GLOBAL_H_ */