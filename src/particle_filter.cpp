/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 * 		Modified: Grigorii Antiokh
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
#include <map>

#include "particle_filter.h"
#include "helper_functions.h"

const double EPSILON = 0.000001;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	m_num_particles = 100;
	std::default_random_engine gen;
	std::normal_distribution<double> dist_x(x, std[0]);
	std::normal_distribution<double> dist_y(y, std[1]);
	std::normal_distribution<double> dist_theta(theta, std[2]);
	for(size_t i = 0; i < m_num_particles; ++i)
	{
		double sample_x = dist_x(gen);
		double sample_y = dist_y(gen);
		double sample_theta = dist_theta(gen);

		m_particles.emplace_back(i, sample_x, 
			sample_y, sample_theta, 1.);
	}

	// std::cout << "=====================" << std::endl;
	// std::cout << "Estimated position:" << std::endl;
	// std::cout << x << " " << y << " " << theta << std::endl;
	// std::cout << "stdevs: " << std[0] << " " << std[1] << " " << std[2] << std::endl;
	// for(const auto& part : m_particles)
	// {
	// 	std::cout << part.id << " " << part.x << " " << 
	// 		part.y << " " << part.theta << " " << part.weight << std::endl;
	// }
	// std::cout << "=====================" << std::endl;

	m_is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	std::default_random_engine gen;
	std::normal_distribution<double> dist_x(0., std_pos[0]);
	std::normal_distribution<double> dist_y(0., std_pos[1]);
	std::normal_distribution<double> dist_theta(0., std_pos[2]);

	double yaw_rate_t = yaw_rate * delta_t;

	for(auto& particle : m_particles)
	{
		double dir = particle.theta + yaw_rate_t;
		if(fabs(yaw_rate) < EPSILON)
		{
			particle.x += velocity * delta_t * cos(particle.theta);
			particle.y += velocity * delta_t * sin(particle.theta);
		}
		else 
		{
			double quotient = velocity / yaw_rate;
			
			particle.x += quotient * (sin(dir) - sin(particle.theta));
			particle.y += quotient * (cos(particle.theta) - cos(dir));
			particle.theta += yaw_rate_t;
		}

		particle.x += dist_x(gen);
		particle.y += dist_y(gen);
		particle.theta += dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::map<int, LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	for(auto& observation : observations)
	{
		auto min_dist = std::numeric_limits<double>::max();

		int id = -1;
		for(const auto& prediction : predicted)
		{
			const auto& pred_id = prediction.first;
			const auto& pred = prediction.second;
			auto distance = dist(observation.x, observation.y, 
				pred.x, pred.y);
			if(distance < min_dist)
			{
				min_dist = distance;
				id = pred_id;
			}
		}
		observation.id = id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	auto normalizer = 1./(2 * M_PI * std_landmark[0] * std_landmark[1]);
	auto doubled_std_landmark_x_2 = 2 * std_landmark[0] * std_landmark[0];
	auto doubled_std_landmark_y_2 = 2 * std_landmark[1] * std_landmark[1];
	
	m_weights.clear();
	m_max_weight = std::numeric_limits<double>::min();

	for(auto& particle : m_particles)
	{
		std::map<int, LandmarkObs> landmarks_in_range;
		for(auto& landmark : map_landmarks.landmark_list)
		{
			if(dist(particle.x, particle.y, landmark.x_f, landmark.y_f) <= sensor_range)
			{
				auto id = landmark.id_i;
				auto x = landmark.x_f;
				auto y = landmark.y_f;
				landmarks_in_range.emplace(id, LandmarkObs{id, x, y});
			}
		}

		if(!observations.empty())
		{
			std::vector<LandmarkObs> obs_transformed;
			for (const auto &observation : observations)
			{
				auto transformed_x = cos(particle.theta) * observation.x - sin(particle.theta) * observation.y + particle.x;
				auto transformed_y = sin(particle.theta) * observation.x + cos(particle.theta) * observation.y + particle.y;
				auto id = observation.id;
				obs_transformed.emplace_back(LandmarkObs{id, transformed_x, transformed_y});
			}

			dataAssociation(landmarks_in_range, obs_transformed);

			double res_weight = 1.;
			bool found = false;

			for (auto &observation : obs_transformed)
			{
				auto found_obs = landmarks_in_range.find(observation.id);
				if (found_obs != std::end(landmarks_in_range))
				{
					found = true;
					auto dx = observation.x - found_obs->second.x;
					auto dy = observation.y - found_obs->second.y;

					auto weight = normalizer * exp(-(dx * dx / doubled_std_landmark_x_2 + 
						dy * dy / doubled_std_landmark_y_2)) + EPSILON;
					res_weight *= weight;
				}
			}
			if(found)
			{
				particle.weight = res_weight;
				m_weights.emplace_back(res_weight);
				if (res_weight > m_max_weight)
				{
					m_max_weight = res_weight;
				}
			}
		}
		else
		{
			std::cout << "No observations provided!" << std::endl;
		}
	}
}

void ParticleFilter::resample() {
	std::default_random_engine gen;
	std::uniform_real_distribution<double> double_dist(0., m_max_weight);
  	std::uniform_int_distribution<int> int_dist(0, m_num_particles - 1);

	auto index = int_dist(gen);

	double beta = 0.0;

	std::vector<Particle> resampled;
	for(size_t i = 0; i < m_num_particles; ++i)
	{
		beta += double_dist(gen) * 2.0;
		while (beta > m_weights[index])
		{
			beta -= m_weights[index];
			index = (index + 1) % m_num_particles;
		}
		resampled.emplace_back(m_particles[index]);
	}

	m_particles = resampled;
}

void ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates
	
    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

std::string ParticleFilter::getAssociations(Particle best)
{
	std::vector<int> v = best.associations;
	std::stringstream ss;
    std::copy( v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
    std::string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

std::string ParticleFilter::getSenseX(Particle best)
{
	std::vector<double> v = best.sense_x;
	std::stringstream ss;
    std::copy( v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
    std::string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

std::string ParticleFilter::getSenseY(Particle best)
{
	std::vector<double> v = best.sense_y;
	std::stringstream ss;
    std::copy( v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
    std::string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
