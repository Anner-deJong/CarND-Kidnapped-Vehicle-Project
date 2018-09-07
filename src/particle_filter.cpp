/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
  //   x, y, theta and their uncertainties from GPS) and all weights to 1. 
  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  
  // hyperparameter: number of particles
  num_particles = 100;
  
  // Create Gaussian distributions for each of the three position variables
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_th(theta, std[2]);

  for (int i=0; i < num_particles; ++i) {
    
    // Create new particle
    Particle particle;
    particle.id = i;
    particle.x  = dist_x(gen);
    particle.y  = dist_y(gen);
    particle.theta = dist_th(gen);
    particle.weight = 1; // keeping track of the weights inside the vector 'weight' is enough actually

    // Add particle to the particles member variable
    particles.push_back(particle);

    // keep track of particles' weights
    weights.push_back(1);
  }

  // Initialize only once
  is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  // TODO: Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/
  
  // Creating Gaussian distributions for each of the three position variables
  // Reinitializing the same distributions in each loop seems somewhat inefficient..
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_th(0, std_pos[2]);
  
  // for readability
  double v = velocity;

  // update location prediction and include position noise
  // process is different yaw_rate is 0
  if (yaw_rate > 0.001 || yaw_rate < -0.001) {
    for (auto &p: particles) {
	  
      // make sure to update p.x and p.y before p.theta                              // noise addition
      p.x     += v/yaw_rate * ( sin(p.theta + yaw_rate*delta_t) - sin(p.theta))      +  dist_x(gen);
      p.y     += v/yaw_rate * (-cos(p.theta + yaw_rate*delta_t) + cos(p.theta))      +  dist_y(gen);
      p.theta += delta_t * yaw_rate                                                  + dist_th(gen);
	}
  }
  else {
    for (auto &p: particles) {
      
	  // make sure to update p.x and p.y before p.theta   // noise addition
      p.x     += v * cos(p.theta) * delta_t               +  dist_x(gen);
      p.y     += v * sin(p.theta) * delta_t               +  dist_y(gen);
      p.theta +=                                            dist_th(gen);
	}
  }

}

LandmarkObs ParticleFilter::dataAssociation(const LandmarkObs &observation, const std::vector<LandmarkObs> &predicted) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
  // std::cout << "getting into dataAssociation" << std::endl;
  // std::cout << "obs.x: " << observation.x, ", obs.y: " << observation.y << std::endl;

	// assign first prediction landmark as the closest landmark
	double min_distance = dist(observation.x, observation.y, predicted[0].x, predicted[0].y);
	LandmarkObs min_dist_pred = predicted[0];

	// loop over all predictions
	for (auto &pred: predicted) {

		double distance   = dist(observation.x, observation.y, pred.x, pred.y);

		if (distance < min_distance) {

			min_distance  = distance;
			min_dist_pred = pred;
		}

	}

// std::cout << "closest chosen prediction" << std::endl;
// std::cout << "pred.x: " << min_dist_pred.x, ", pred.y: " << min_dist_pred.y << std::endl;

	return min_dist_pred;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

  // NOTE 1: no need to normalize for the std::discrete_distribution function
  // get the sum of the weights for normalizing
  // double weights_sum = 0;

  // for each particle, calculate the new weight
	for (int i=0; i<particles.size(); ++i){

    // for readability
    Particle &p = particles[i];

		// get all map landmarks within sensor range
		std::vector<LandmarkObs> predicted;

		for (auto &lm: map_landmarks.landmark_list) {
			if (dist(p.x, p.y, lm.x_f, lm.y_f) < sensor_range) {
				LandmarkObs pred;
			    pred.id = lm.id_i;
			    pred.x  = lm.x_f;
			    pred.y  = lm.y_f;
			    predicted.push_back(pred);
			}
		}

    // this will become the updated weight
    double new_weight = 1;
		
    // get partial weight for each observation:
		for (auto &ob: observations) {

			// transform observation into map coordinates
			LandmarkObs tobs;
			tobs.x = cos(p.theta)*ob.x - sin(p.theta)*ob.y + p.x;
			tobs.y = sin(p.theta)*ob.x + cos(p.theta)*ob.y + p.y;

			// get closest prediction
			LandmarkObs closest_pred = dataAssociation(tobs, predicted);

			// use multi variate gaussian density function to update weights
			new_weight *= MultivariateGaussian_Landmarks(tobs, closest_pred, std_landmark);
		}

    // NOTE 1: no need to normalize for the std::discrete_distribution function
		// update the particle's weight, and keep track of the sum
		// keeping track of weights inside the member variable vector weights is enough actually
    p.weight   = new_weight;
    // weights_sum += new_weight;
    weights[i] = new_weight;

	}

  // NOTE 1: no need to normalize for the std::discrete_distribution function
  // saving an extra for loop over all particles / weight entries
	// normalize weights into sum=1
  // for (int i=0; i<particles.size(); ++i){
  //   particles[i].weight /= weights_sum;
  //   weights[i] = particles[i].weight;
  // }

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  
  // this distribution doesn't actually require normalized weights
  std::discrete_distribution<> distr(weights.begin(), weights.end());

  // // build up new vector of particles by sampling through the distribution
  std::vector<Particle> resampled_particles;
  for (int i=0; i<num_particles; ++i) {
    resampled_particles.push_back(particles[distr(gen)]);
  }

  // make the new resampled vector the current one
  particles = resampled_particles;

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    return particle; // this was commented out which results in a warning.
    // no error however so this function is likely not called
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
