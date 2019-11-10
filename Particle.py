
from PSOPopulation import PSO
import random

class Particle(PSO):
	
	def __init__(self, pos, vel, index):
		self.num_of_param = len(pos)
		self.p_best_pos = pos
		self.position = pos
		self.velocity = vel
		self.p_best_cost = None
		self.p_cost = None
		self.id = index

	def calculatePosition(self):
		for i in range(self.num_of_param):
			self.position[i] = self.position[i] + self.velocity[i]
			self.position[i] = max(self.position[i], PSO.position_min[i])
			self.position[i] = min(self.position[i], PSO.position_max[i])

	def calculateVelocity(self):
		for i in range(self.num_of_param):
			self.velocity[i] = self.w * self.velocity[i] + self.c1 * random.uniform(0,1) * (self.p_best_pos[i] - self.position[i]) + self.c2 * random.uniform(0,1) * (PSO.g_best_particle.p_best_pos[i] - self.position[i])
			self.velocity[i] = max(self.velocity[i], PSO.velocity_min[i])
			self.velocity[i] = min(self.velocity[i], PSO.velocity_max[i])

	def setGBest(self):
		PSO.g_best_particle = self

	def setPBest(self, pos, value):
		self.p_best_pos = pos.copy()
		self.p_best_cost = value