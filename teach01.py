import random
import decimal


class Movie:
    def __init__(self, title, year, runtime):
        self.title = title
        self.year = year
        if (runtime > 0):
            self.runtime = runtime
        else:
            self.runtime = 0

    def __repr__(self):
        return self.title + " (" + str(self.year) + ") - " + str(self.runtime) + " mins."

    def to_hours(self):
        return self.runtime / 60, self.runtime % 60


movie = Movie("Jurassic Park", 2015, 124)

print(movie)
print(movie.to_hours())


def create_movie_list():
    my_list = [Movie("Star Wars 1", 2000, 100),
               Movie("Star Wars 2", 2002, 125),
               Movie("Star Wars 3", 2004, 150),
               Movie("Star Wars 4", 2006, 175),
               Movie("Star Wars 5", 2008, 200)]
    return my_list


def main():
    my_list = create_movie_list()
    for i in my_list:
        print(i)

    print("\nMovies with a runtime of longer than 150 minutes:")
    new_list = [i for i in my_list if i.runtime > 150]

    for i in new_list:
        print(i)

    dict = {}
    for i in my_list:
        dict[i.title] = decimal.Decimal(random.randint(0, 500))/100

    for i in dict:
        print(i + " - " + str(dict[i]))


if __name__ == '__main__':
    main()


