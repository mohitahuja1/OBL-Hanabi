import copy
import random

def get_hand_probas(observation, belief_hand):
	"""Claculate hand probabilities (belief model -	level 0)
	Args:
		fireworks
		observed_hands
		discard_pile
		card_knowledge
	Returns:
		probas: list, For each card in hand, probability of it belonging to one of the 25 
			possiblecards. Contains 125 elements (25 probas for each of the 5 cards in hand)
	"""
	
# 	current_player = observations['current_player']
# 	observation = (observations['player_observations'][current_player])
	str_pyhanabi = str(observation['pyhanabi'])
	fireworks_pos = str_pyhanabi.find("Fireworks:")
	hands_pos = str_pyhanabi.find("Hands:")
	fireworks_str = str_pyhanabi[fireworks_pos + 11:hands_pos-2]
	fireworks_str_list = fireworks_str.split()
	fireworks_dict = {}
	for e in fireworks_str_list:
			fireworks_dict[e[0]] = int(e[1])
	# Extract 4 relevant variables from observation
	fi = copy.deepcopy(fireworks_dict)
	oh = observation['observed_hands']
	dp = observation['discard_pile']
	ck	= observation['card_knowledge']

	# Pre process the input

	fireworks = [fi['R'], fi['Y'], fi['G'], fi['W'], fi['B']]

	# print("fi: ", fi)
	# print("fireworks: ", fireworks)

	observed_hands = []
	if len(oh) > 0:
		for e in oh[1]:
				observed_hands.append(e['color'] + str(e['rank']))

	# print("oh: ", oh)
	# print("observed_hands: ", observed_hands)

	discard_pile = []
	if len(dp) > 0:
		for e in dp:
			discard_pile.append(e['color'] + str(e['rank']))

	# print("dp: ", dp)
	# print("discard_pile: ", discard_pile)

	# Create a list of all visible cards in the game
	visible_cards = []
	colors = 'RYGWB'

	# Add all possible cards from the fireworks list
	for i, j in enumerate(fireworks):
		j = int(j)
		if j > 0:
			for k in range(0, j):
				visible_cards.append(colors[i] + str(k))

	# Add team mates hand and discard pile to visible cards
	visible_cards.extend(observed_hands)
	visible_cards.extend(discard_pile)

	# print("visible_cards: ", visible_cards)

	# Create a string representation of each possible card that contains the color, rank, and possible count
	probas = []
	positions = ["R0_3", "R1_2", "R2_2", "R3_2", "R4_1", "Y0_3", "Y1_2", "Y2_2", "Y3_2", "Y4_1", "G0_3", "G1_2", "G2_2", "G3_2", "G4_1", "W0_3", "W1_2", "W2_2", "W3_2", "W4_1", "B0_3", "B1_2", "B2_2", "B3_2", "B4_1"]

	# Use hints in card knowledge to gather possible ranks and colors for each position

	for e in ck[0]:
		if e['color'] is None:
			colors = 'RYGWB'
		else:
			colors = e['color']
		if e['rank'] is None:
			ranks = '01234'
		else:
			ranks = str(e['rank'])

		l2 = []
		# Check how many cards are possible for each position based on hints and visible cards
		for position in positions:
			card = position.split("_")[0]
			possibles = int(position.split("_")[1])
			if card[0] in colors and card[1] in ranks:
				# print("possibles: ", possibles)
				# print("vc: ", visible_cards.count(card))
				# print("visible_cards: ", visible_cards)
				l2.append(possibles - visible_cards.count(card))
			else:
				l2.append(0)

		# print("ck[0]: ", ck[0])
		# print("colors: ", colors)
		# print("ranks: ", ranks)
		# print("visible_cards: ", visible_cards)

		# print("l2: ", l2)
		# Caclulate probabilities based on possible card counts
		probas.extend([e/sum(l2) for e in l2])
	missing_cards_len = 125 - len(probas)
	print("missing_cards_len: ", missing_cards_len)
	probas.extend([0]*missing_cards_len)
	print("probas")
	probas_chunks = chunks_fn(probas, 25)
	print("probas_chunks: ", probas_chunks)
	samples = []
	for chunk in probas_chunks:
		print("chunk: ", chunk)
		m = max(chunk)
		print("m: ", m)
		max_indices = [u for u, v in enumerate(chunk) if v == m]
		print("max_indices: ", max_indices)
		max_index = random.choice(max_indices)
		print("max_index: ", max_index)
		for i, j in enumerate(chunk):
			if i == max_index and j > 0:
				samples.append(1)
			else:
				samples.append(0)
	print("belief probas: ", probas)
	print("belief samples: ", samples)
	if belief_hand == 0:
		hand = probas
	elif belief_hand == 1:
		hand = samples
	return hand
	
def chunks_fn(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]