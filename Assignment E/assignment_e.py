from colorsys import hls_to_rgb
from pathlib import Path
from PIL import Image, ImageDraw
from random import randint, sample, shuffle
from sys import argv
from typing import Callable, NamedTuple

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


def input_or_default(prompt: str, default: int) -> int:
    try:
        return int(input(prompt))
    except ValueError:
        return default


def random_genome(num_cities: int) -> Genome:
    genome = list(range(num_cities)) * 2
    shuffle(genome)
    return genome


def init_population(size: int, num_cities: int) -> list[Genome]:
    return [random_genome(num_cities) for _ in range(size)]


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


def swap(list: list, i: int, j: int):
    temp = list[i]
    list[i] = list[j]
    list[j] = temp


def mutate(genome: Genome):
    num_swaps = randint(1, 10)
    for _ in range(num_swaps):
        i, j = sample(range(len(genome)), 2)
        swap(genome, i, j)


def make_child(parents: list[Genome]) -> Genome:
    parent_1, parent_2 = sample(parents, 2)
    child = recombination_simple_crossover(parent_1, parent_2)
    mutate(child)
    return child


def offspring(parents: list[Genome], lmbda: int) -> list[Genome]:
    return [make_child(parents) for _ in range(lmbda)]


def ea_step(population: list[Genome], fitness_func: Callable[[Genome], float], mu: int) -> list[Genome]:
    population = population[:mu] + offspring(population[:mu], len(population) - mu)
    population.sort(key=fitness_func, reverse=True)
    return population


def print_performance(population: list[Genome], fitness_func: Callable[[Genome], float], i: int):
    print(f"It. {i:3} - Best: {fitness_func(population[0]):.2f}" +
          f", Median: {fitness_func(population[len(population) // 2]):.2f}" +
          f", Worst: {fitness_func(population[-1]):.2f}")


def hsla_to_rgba(h:float, s:float, l:float, a:float):
    r, g, b = hls_to_rgb(h, l, s)
    return tuple(int(c * 255) for c in [r, g, b, a])


def viz_path(path: Genome, cities: list[City], img_path: Path):
    map_image = Image.open(img_path)
    draw = ImageDraw.Draw(map_image)

    total = len(path) - 1
    for i in range(total):
        coords_1 = (cities[path[i]].pos_x, cities[path[i]].pos_y)
        coords_2 = (cities[path[i+1]].pos_x, cities[path[i+1]].pos_y)
        hue = i / total
        rgba = hsla_to_rgba(hue, 0.8, 0.5, 0.5)
        draw.line([coords_1, coords_2], fill=rgba, width=3)

    # draw circle around start and end
    coords_start = (cities[path[0]].pos_x, cities[path[0]].pos_y)
    coords_end = (cities[path[-1]].pos_x, cities[path[-1]].pos_y)
    r = 10
    draw.ellipse((cities[path[0]].pos_x - r, cities[path[0]].pos_y - r,
                 cities[path[0]].pos_x + r, cities[path[0]].pos_y + r),
                 outline="red", width=3)
    draw.ellipse((cities[path[-1]].pos_x - r, cities[path[-1]].pos_y - r,
                 cities[path[-1]].pos_x + r, cities[path[-1]].pos_y + r),
                 outline="red", width=3)

    
    map_image.save(f"Solution (fitness={fitness(path, cities):.0f}).png")


def print_path(path: Genome, cities: list[City]):
    print(" -> ".join([cities[c].name for c in path]))


def main():
    if len(argv) != 3:
        raise ValueError(f"Usage: {argv[0]} <path_to_cites_file> <path_to_map>")

    cities = parse(Path(argv[1]))
    fitness_func = lambda gen: fitness(gen, cities)

    p = input_or_default("Choose population size (P, default 100): ", 100)
    mu = input_or_default("Choose size of parent pool (mu, default 20): ", 20)
    if mu > p:
        raise ValueError("mu must be <= P")
    # num_iter = input_or_default("Choose number of iterations (default 100): ", 100)
    
    population = init_population(p, len(cities))
    population.sort(key=fitness_func, reverse=True)

    print_performance(population, fitness_func, 0)
    # for i in range(num_iter):
    try:
        i = 1
        while True:
            population = ea_step(population, fitness_func, mu)
            print_performance(population, fitness_func, i)
            i += 1
    except KeyboardInterrupt:
        ...

    best = population[0]
    viz_path(best, cities, Path(argv[2]))
    print("\nBest path:")
    print_path(best, cities)
    print("Path is valid.") if is_valid(best, cities) else print("your god damn code is ass")


if __name__ == "__main__":
    main()