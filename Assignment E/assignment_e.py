from pathlib import Path
from random import randint
from sys import argv
from typing import NamedTuple

class City(NamedTuple):
    id: int  # not really needed, but it's there and it doesn't hurt
    name: str
    pos_x: int
    pos_y: int


Genome = list[int]  # list of indices of cities


def parse(file: Path) -> list[City]:
    if not file.exists() or not file.is_file():
        raise ValueError(f"File {file.absolute()} not found.")

    cities = []
    with open(file) as f:
        for line in f.readlines():
            if line.startswith('#'):
                continue
            id, pos_x, pos_y, name = map(str.strip, line.split('\t'))
            cities.append(City(int(id), name[1:], int(pos_x), int(pos_y)))
    return cities


def distance(a: City, b: City) -> float:
    dx = b.pos_x - a.pos_x
    dy = b.pos_y - a.pos_y
    return (dx * dx + dy * dy) ** 0.5


def fitness(genome: Genome, cities: list[City]) -> float:
    score = 0
    for i in range(len(genome) - 1):
        score += distance(cities[genome[i]], cities[genome[i+1]])
    return score


def is_valid(genome: Genome, cities: list[City]) -> bool:
    if len(genome) != 2 * len(cities):
        return False
    
    for i in range(len(cities)):
        if genome.count(i) != 2:
            return False
        
    return True


def recombination_simple_crossover(parent_1: Genome, parent_2: Genome) -> Genome:
    crossover = randint(0, len(parent_1) - 1)

    offspring = parent_1[:crossover]  # take everything up to crossover from parent 1
    city_count = [0] * (len(parent_1) // 2)

    for city in offspring:
        city_count[city] += 1

    for city in parent_2:  # for the rest: take remaining cities in the order they appear in parent 2
        if city_count[city] == 2:
            continue
        offspring.append(city)
        city_count[city] += 1

    return offspring


def main():
    if len(argv) != 2:
        raise ValueError(f"Usage: {argv[0]} <path_to_cites_file>")
    cities = parse(Path(argv[1]))
    for city in cities:
        print(city)


if __name__ == "__main__":
    main()