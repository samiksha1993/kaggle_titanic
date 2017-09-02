def child(row):
	if row["Age"] < 12:
		return 1
	else:
		return 0

def senior_citizen(row):
	if row["Age"] > 60:
		return 1
	else:
		return 0

def female_married(row):
	if row["Sex"] == 0:
		if len(row["Name"].split('('))>0:
			return 1
	return 0