import random

class PSO:

	# Variaveis coeficientes PSO
	c1 = 2
	c2 = 2
	w = 0.7
	
	g_best_particle = None
	
	population = []
	
	# Posição Limite
	position_min = []
	position_max = []

	# Velocidade Limite
	velocity_min = []
	velocity_max = []
	
	def __init__(self, pos_min, pos_max):
		#Initializing Position Min and Max
		PSO.position_min = pos_min
		PSO.position_max = pos_max
		#Initializing Velocity Min and Max
		for index in range(len(self.position_min)):
			PSO.velocity_min.append(-0.1 * (PSO.position_max[index] - PSO.position_min[index]))
			PSO.velocity_max.append(0.1 * (PSO.position_max[index] - PSO.position_min[index]))
		
	def initializeParticle(self, index):
		from Particle import Particle
		pos = []
		for i in range(len(PSO.position_min)):
			if PSO.position_min[i] == int(PSO.position_min[i]):
				aux = random.randrange(PSO.position_min[i],PSO.position_max[i]) # 0
			else:
				aux = random.uniform(PSO.position_min[i],PSO.position_max[i]) # 2
			pos.append(aux)

		vel = []
		for i in range(len(self.velocity_min)):
			if PSO.velocity_min[i] == int(PSO.velocity_min[i]):
				aux = random.randrange(PSO.velocity_min[i],PSO.velocity_max[i]) # 0
			else:
				aux = random.uniform(PSO.velocity_min[i],PSO.velocity_max[i]) # 2
			vel.append(aux)

		self.population.append(Particle(pos, vel, index))
		
	def insertParticleCost(self, index, cost):
		PSO.population[index].p_cost = cost
		
		if PSO.population[index].p_best_cost == None or  PSO.population[index].p_best_cost < cost:
			PSO.population[index].setPBest(PSO.population[index].position, cost)
		
		if PSO.g_best_particle == None or cost > PSO.g_best_particle.p_best_cost:
			PSO.population[index].setGBest()
			
	def getPosition(self, index):
		return self.population[index].position

	def printGlobalBestParticle(self):
		print('-----------------------------------------------')
		print('Best Particle: {}'.format(self.g_best_particle.id))
		print('Best Position:' + str(self.g_best_particle.p_best_pos))
		print('Best Cost: {}'.format(self.g_best_particle.p_best_cost))
		print('-----------------------------------------------')
		
	def printPosition(self):
		print('Position Min: ' + str(self.position_min))
		print('Position Max: ' + str(self.position_max))
		
	def printVelocity(self):
		print('Velocity Min: ' + str(self.velocity_min))
		print('Velocity Max: ' + str(self.velocity_max))