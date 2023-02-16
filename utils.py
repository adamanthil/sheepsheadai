from collections import namedtuple


PlayerExperience = namedtuple(
	"PlayerExperience",
	field_names=[
		"state",
		"action",
		"reward",
		"next_state",
		"done",
	]
)
