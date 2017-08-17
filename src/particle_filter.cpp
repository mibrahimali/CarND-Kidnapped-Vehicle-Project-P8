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
	default_random_engine gen;
	double std_x, std_y, std_psi; // Standard deviations for x, y, and psi
	this->num_particles = 500;
	//Set standard deviations for x, y, and psi
	 std_x = std[0];
	 std_y = std[1];
	 std_psi = std[2];
	 
	// This line creates a normal (Gaussian) distribution for particles pose
	normal_distribution<double> dist_x(x, std_x);
	normal_distribution<double> dist_y(y, std_y);
	normal_distribution<double> dist_psi(theta, std_psi);

	for (int i = 0; i < this->num_particles; i++) {
		Particle particle_i;
		particle_i.x = dist_x(gen);
		particle_i.y = dist_y(gen);
		particle_i.theta = dist_psi(gen);	 
		particle_i.weight = 1;
		this->particles.push_back(particle_i);
		this->weights.push_back(1);
		// for debuging purpose
		// cout<<"my particle "<< i<< "initialized at x= "<<particle_i.x<<" y = " << particle_i.y <<" theta = "<< particle_i.theta <<endl;
	}
	this->is_initialized = true;
	
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine gen;
	double std_x, std_y, std_psi; // Standard deviations for x, y, and psi
	//Set standard deviations for x, y, and psi
	 std_x = std_pos[0];
	 std_y = std_pos[1];
	 std_psi = std_pos[2];
	double new_x,new_y,new_theta;
	

	for (int i = 0; i < this->num_particles; i++) {
		// update each particle w.r.t motion model accounting for motion noise
		// check if yaw rate is zero
		if (fabs(yaw_rate) < 0.00001)
		{
			new_x = this->particles[i].x + velocity * delta_t * cos(this->particles[i].theta);
			new_y = this->particles[i].y + velocity * delta_t * sin(this->particles[i].theta);
			new_theta = this->particles[i].theta;
		}
		else
		{
			new_x = this->particles[i].x + (velocity/yaw_rate) *(sin(this->particles[i].theta + yaw_rate*delta_t) - sin(this->particles[i].theta));
			new_y = this->particles[i].y + (velocity/yaw_rate) *(cos(this->particles[i].theta) - cos(this->particles[i].theta + yaw_rate*delta_t));
			new_theta = this->particles[i].theta + yaw_rate*delta_t;
		}
		// This line creates a normal (Gaussian) distribution for prediction step noise 
		normal_distribution<double> dist_x(new_x, std_x);
		normal_distribution<double> dist_y(new_y, std_y);
		normal_distribution<double> dist_psi(new_theta, std_psi);
		this->particles[i].x = dist_x(gen);
		this->particles[i].y = dist_y(gen);
		this->particles[i].theta  = dist_psi(gen);
		// for debuging purpose
		// cout<<"my predected particle "<< i<< " at x= "<<this->particles[i].x<<" y = " << this->particles[i].y <<" theta = "<< this->particles[i].theta <<endl;
		// cout << "using velocity = "<< velocity<<"and yaw rate = "<<yaw_rate<<endl;
	}
}


std::vector<LandmarkObs> ParticleFilter::predictObservation(Particle particle, double sensor_range, Map map_landmarks){
	double x_p = particle.x;
	double y_p = particle.y;
	double measured_dist;
	std::vector<LandmarkObs> predicted_observations;
	for(int landmark_id=0; landmark_id<map_landmarks.landmark_list.size();landmark_id++){
	measured_dist = dist(x_p,y_p,map_landmarks.landmark_list[landmark_id].x_f,map_landmarks.landmark_list[landmark_id].y_f);
	if(measured_dist <= sensor_range)
	{
		LandmarkObs predicted_landmark;
		predicted_landmark.id = map_landmarks.landmark_list[landmark_id].id_i;
		predicted_landmark.x = map_landmarks.landmark_list[landmark_id].x_f;
		predicted_landmark.y = map_landmarks.landmark_list[landmark_id].y_f;

		predicted_observations.push_back(predicted_landmark);
	}
	}
	return predicted_observations;

}


std::vector<LandmarkObs> ParticleFilter::transformObservation(Particle particle, std::vector<LandmarkObs>& observations){
	double x_p = particle.x;
	double y_p = particle.y;
	double theta_p = particle.theta;
	double x_m,y_m;
	std::vector<LandmarkObs> transformed_observations;
	for(int observation=0; observation<observations.size();observation++){
		LandmarkObs transformed_observation;
		transformed_observation.id = -1;
		transformed_observation.x = x_p + (cos(theta_p)*observations[observation].x) - (sin(theta_p)*observations[observation].y);
		transformed_observation.y = y_p + (sin(theta_p)*observations[observation].x) + (cos(theta_p)*observations[observation].y);
		transformed_observations.push_back(transformed_observation);
	}
	return transformed_observations;
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	double best_association_dist;
	for (int observation_id=0;observation_id<observations.size();observation_id++)
		{
			best_association_dist = std::numeric_limits<double>::infinity();
			for (int prediction_id=0;prediction_id<predicted.size();prediction_id++)
			{
				double distance = dist(observations[observation_id].x,observations[observation_id].y,
																predicted[prediction_id].x,predicted[prediction_id].y);

				if(distance <best_association_dist)
				{
					best_association_dist = distance;
					observations[observation_id].id = predicted[prediction_id].id;
				}
			}
		}


}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
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

	double std_landmark_x = std_landmark[0];
	double std_landmark_y = std_landmark[1];
	for (int i = 0; i < this->num_particles; i++) {
		std::vector<LandmarkObs> predicted_observations = this->predictObservation(this->particles[i], sensor_range, map_landmarks);
		std::vector<LandmarkObs> transformed_observations = this->transformObservation(this->particles[i],observations);
		this->dataAssociation(predicted_observations,transformed_observations);
		double weight = 1 ;
		std::vector<int> associations; std::vector<double> sense_x; std::vector<double> sense_y;
		for (int j=0;j<transformed_observations.size();j++)
		{
			
			int assoicated_map_landmark = transformed_observations[j].id -1;
			double single_measure_prob = multi_var_gaussian_calculater(transformed_observations[j].x, transformed_observations[j].y,
									 map_landmarks.landmark_list[assoicated_map_landmark].x_f, map_landmarks.landmark_list[assoicated_map_landmark].y_f,
									  std_landmark_x, std_landmark_y);
			weight *=	single_measure_prob;

			associations.push_back(transformed_observations[j].id);
			sense_x.push_back(transformed_observations[j].x);
			sense_y.push_back(transformed_observations[j].y);
		}
		
		this->particles[i] = SetAssociations(this->particles[i],associations,sense_x,sense_y);
		this->particles[i].weight = weight;
		this->weights[i] = weight;
		// cout<<"particle id"<< i << "has weight = "<<this->particles[i].weight<<endl;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	std::vector<Particle> resampled_particles;
	std::vector<double> resampled_weights;
	std::default_random_engine generator;
  std::discrete_distribution<int> distribution(this->weights.begin(), this->weights.end());
	for (int i = 0; i < this->num_particles; i++) {
		int particle_id = distribution(generator);

		Particle particle_i;
		particle_i= this->particles[particle_id];
		resampled_particles.push_back(particle_i);
		resampled_weights.push_back(this->particles[particle_id].weight);
	}
	this->particles.clear();
	this->weights.clear();
	this->particles = resampled_particles;
	this->weights = resampled_weights;

}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
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
