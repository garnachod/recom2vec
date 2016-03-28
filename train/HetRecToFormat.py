import json

if __name__ == '__main__':

	users = {}
	with open("test", "r") as fIn:
		for i, line in enumerate(fIn):
			if i == 0:
				pass
			else:
				lineParsed = line.replace("\n", "").split("\t")
				user = lineParsed[0]
				movie = lineParsed[1]
				rating = float(lineParsed[2])
				timestamp = int(lineParsed[3])

				if user not in users:
					users[user] = {"ratings" : [], "average": 0.0, "ratingsCount": 0}

				users[user]["ratings"].append((movie, rating, timestamp))
				users[user]["average"] += rating
				users[user]["ratingsCount"] += 1

	#get the real average and sort ratings by timestamp
	for user in users:
		users[user]["average"] = users[user]["average"] / users[user]["ratingsCount"]
		users[user]["ratings"] = sorted(users[user]["ratings"], key=lambda x: x[2], reverse=False)

	#write out file
	with open("ratings_parsed_test.txt", "w") as fOut:
		fOut.write(json.dumps(users))
