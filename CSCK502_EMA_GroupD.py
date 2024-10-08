# script to run algorithms in GUI. scrabble_words.txt must be in the same folder


import random  # Import the random module for generating random numbers
import tkinter as tk  # Import tkinter module for creating GUI applications
from tkinter import messagebox  # Import messagebox for displaying error messages
from tkinter import scrolledtext #Import scrollbox for output box
import pandas as pd #pandas is needed for handling the board and solution as DataFrame
import numpy as np #numpy needed for manipulating lists and reshaping

# Function to create a random grid based on an initial grid and available letters
def random_grid(initial_grid, LETTERS):
    grid = []  # Initialize an empty list to hold the new grid
    for row in initial_grid:  # Iterate over each row in the initial grid
        new_row = []  # Initialize a new row for the grid
        for letter in row:  # Iterate over each letter in the row
            if letter == '':  # If the cell is empty
                new_row.append(random.choice(LETTERS))  # Randomly choose a letter from LETTERS
            else:
                new_row.append(letter)  # Keep the fixed letter
        grid.append(new_row)  # Append the new row to the grid
    return grid  # Return the completed grid

# Function to evaluate the fitness of a grid
def evaluate_fitness(grid):
    score = 0  # Initialize score
    GRID_SIZE = 4  # Define the grid size
    for i in range(GRID_SIZE):  # Iterate over each row index
        # Rows:
        if len(set(grid[i])) == GRID_SIZE:  # Check if all letters in the row are unique
            score += 1  # Increment score for unique row

        # Columns:
        if len(set([grid[j][i] for j in range(GRID_SIZE)])) == GRID_SIZE:  # Check column uniqueness
            score += 1  # Increment score for unique column

    # Sub-Grids:
    for x in range(0, GRID_SIZE, 2):  # Iterate over the starting row of 2x2 sub-grids
        for y in range(0, GRID_SIZE, 2):  # Iterate over the starting column of 2x2 sub-grids
            sub_grid = [grid[x][y], grid[x][y + 1], grid[x + 1][y], grid[x + 1][y + 1]]  # Get the sub-grid
            if len(set(sub_grid)) == GRID_SIZE:  # Check if all letters in the sub-grid are unique
                score += 1  # Increment score for unique sub-grid
    return score  # Return the final score

# Function to perform crossover between two parent grids
def crossover(parent1_grid, parent2_grid, initial_grid):
    child1_grid = []  # Initialize the first child grid
    child2_grid = []  # Initialize the second child grid
    GRID_SIZE = 4  # Define the grid size
    for i in range(GRID_SIZE):  # Iterate over each row index
        new_row1 = []  # Initialize a new row for the first child
        new_row2 = []  # Initialize a new row for the second child
        for j in range(GRID_SIZE):  # Iterate over each column index
            if initial_grid[i][j] == '':  # Only crossover in empty cells
                if random.random() > 0.5:  # Randomly choose between the two parents
                    new_row1.append(parent1_grid[i][j])  # Take from parent1
                    new_row2.append(parent2_grid[i][j])  # Take from parent2
                else:
                    new_row1.append(parent2_grid[i][j])  # Take from parent2
                    new_row2.append(parent1_grid[i][j])  # Take from parent1
            else:
                new_row1.append(initial_grid[i][j])  # Keep the fixed letter
                new_row2.append(initial_grid[i][j])  # Keep the fixed letter
        child1_grid.append(new_row1)  # Append the new row to child1
        child2_grid.append(new_row2)  # Append the new row to child2
    return child1_grid, child2_grid  # Return both child grids

# Function to mutate an individual grid
def mutate(individual_grid, initial_grid):
    GRID_SIZE = 4  # Define the grid size
    row = random.randint(0, GRID_SIZE - 1)  # Randomly select a row to mutate
    non_fixed_cols = [i for i in range(GRID_SIZE) if initial_grid[row][i] == '']  # Get columns that are not fixed
    if len(non_fixed_cols) > 1:  # Ensure there are at least two columns to swap
        col1, col2 = random.sample(non_fixed_cols, 2)  # Randomly select two non-fixed columns
        # Swap the letters in the selected columns
        individual_grid[row][col1], individual_grid[row][col2] = individual_grid[row][col2], individual_grid[row][col1]
        print(f"Mutation occurred: swapped columns {col1} and {col2} in row {row}")  # Print mutation message


# Function to perform the genetic algorithm
def genetic_algorithm(input_grid, output_box):
    initial_grid = [['' for _ in range(4)] for _ in range(4)]  # Create a blank initial grid for fixed positions

    LETTERS = list(set(letter for row in input_grid for letter in row if letter.strip()))  # Get unique letters, ignoring spaces

    population_size = 100  # Define population size
    generations = 1000  # Define number of generations
    mutation_rate = 0.01  # Define mutation rate

    population = [random_grid(input_grid, LETTERS) for _ in range(population_size)]  # Use input_grid to respect fixed letters

    for generation in range(generations):  # Loop through each generation
        population = sorted(population, key=lambda grid: evaluate_fitness(grid), reverse=True)  # Sort population by fitness
        if evaluate_fitness(population[0]) == 12:  # Check if the best grid has maximum fitness
            output_box.insert(tk.END, f"***Solution found in generation {generation + 1}***\n") #dispay in output box
            return population[0]  # Return the solution

        print(f"Here is the best population generated in generation {generation + 1}")
        output_box.insert(tk.END, f"Here are all populations generated in generation {generation + 1}\n")
        print(population)  # Print the best grid in this generation
        output_box.insert(tk.END, f" {population}\n")

        next_population = population[:10]  # Keep the top 10 grids for the next generation
        mutation_count = 0  # Initialize mutation counter
        while len(next_population) < population_size:  # Fill the rest of the population
            parent1, parent2 = random.sample(population[:50], 2)  # Randomly select two parents
            child1, child2 = crossover(parent1, parent2, input_grid)  # Perform crossover
            if random.random() < mutation_rate:  # Randomly decide whether to mutate
                mutate(child1, input_grid)  # Mutate first child
                mutation_count += 1  # Increment mutation count
            if random.random() < mutation_rate:  # Randomly decide whether to mutate
                mutate(child2, input_grid)  # Mutate second child
                mutation_count += 1  # Increment mutation count
            next_population.extend([child1, child2])  # Add children to the next population

        print(f"Number of mutations in generation {generation + 1}: {mutation_count}")  # Print the number of mutations
        population = next_population  # Set the current population to the next generation

    output_box.insert(tk.END, "No solution found after 1000 generations.\n")  # Inform user if no solution is found
    return None

def dist_lett(df_in) : #this function returns all distinct letters frm the player entered board
  #find distinct number of letters
  flatBoard = df_in.to_numpy().flatten() #it is easier to extract the letters consitently as a flattened nump
  nonDupFlatBoard = set(flatBoard) #get rid of duplicates
  nonDupFlatBoard = np.array(list(nonDupFlatBoard)) #and sore this as a numpy array
  return nonDupFlatBoard #before returning

def dist_lett_cnt(df_in) : #we also need to know how many letters there are at several points, so making this a function too
  #find distinct number of letters. Most of this code s identical to dist_lett
  flatBoard = df_in.to_numpy().flatten()
  nonDupFlatBoard = set(flatBoard)
  nonDupFlatBoard = np.array(list(nonDupFlatBoard))
  nonBlankNonDupFlatBoard = nonDupFlatBoard[nonDupFlatBoard != ''] #we don't want to count the space character as a letter
  distinctLetters_cnt = (len(nonBlankNonDupFlatBoard))
  return distinctLetters_cnt

def mkChrom(df_in) :
  i = 0 #we need an itterator, so initialise
  nonDupFlatBoard = dist_lett(df_in) #"nonDupFlatBoard" is the distinct non duplicated letters derived from the defined function dist_lett
  distinctLetters_cnt = dist_lett_cnt(df_in) #we need the number of distinct letters so call dist_lett_cnt function
  blankChrom = '000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000'
  # above is some massive binary zero to chop up. There are neater ways to do this so this might be changed time permitting, but will enable the construction of any ength of chromesome irrespective of number of letters entered by the player (00000 only caters for 4 distinct letters, so if there were 5 then we would have 6*16 == 96 character chromesome)
  blankChrom = blankChrom[:(distinctLetters_cnt+1)] # chop into corrct size for the chromesome based on number of letters
  processingDataframe = df_in # copy the board to holding pandas dataframe
  for letter in nonDupFlatBoard :
    processingDataframe = processingDataframe[processingDataframe != letter] #replace the letter in the copied processingDataframe with null
    chromStr = blankChrom[0:i]+'1'+blankChrom[i+1:] #the string representing whether the square is the letter is 1 for the position through the blankChom. This is NOT the binary number representing the itteration through the letters, i
    processingDataframe = processingDataframe.fillna(chromStr) # replace the blanks with the string representing the letter
    i = i + 1
  chromFlat = processingDataframe.to_numpy().flatten() #easier to analyse the board as a flattened array
  chromBoard = "" #this will be our chromesome returned, so define as string
  for i2 in chromFlat : #itterate through every square in the board
    chromBoard = chromBoard + i2 #concatenate the individual squares binary stings together int one long sting
  return chromBoard #and return

def mkChromBoard(df_in) : # this function is similar to above, but will return the binary character representation of the board rather than the chromesome
  i = 0 #we need an itterator, so initialise
  nonDupFlatBoard = dist_lett(df_in) #"nonDupFlatBoard" is the distinct non duplicated letters derived from the defined function dist_lett
  distinctLetters_cnt = dist_lett_cnt(df_in) #we need the number of distinct letters so call dist_lett_cnt function
  blankChrom = '000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000'
  # above is some massive binary zero to chop up. There are neater ways to do this so this might be changed time permitting, but will enable the construction of any ength of chromesome irrespective of number of letters entered by the player (00000 only caters for 4 distinct letters, so if there were 5 then we would have 6*16 == 96 character chromesome)
  blankChrom = blankChrom[:(distinctLetters_cnt+1)] # chop into corrct size for the chromesome based on number of letters
  a = df_in # copy the board to holding df

  for letter in nonDupFlatBoard :
    a = a[a != letter] #replace the letter in the copied df with null
    chromStr = blankChrom[0:i]+'1'+blankChrom[i+1:] #the string representing whether the square is the letter is 1 for the position through the blankChom. This is not the bindary number representing the itteration through the letters, i
    a = a.fillna(chromStr) # replace the blanks with the string representing the letter
    i = i + 1
  return a

def blankChr(df_in) : #this function returns the "blank character" - the binary representation of a blank space on the board
  i = 0 ############################################################################### the logic is the same as above \/\/\/\/\/\/
  nonDupFlatBoard = dist_lett(df_in)                                                  #
  distinctLetters_cnt = dist_lett_cnt(df_in)                                          #
  blankChrom = '000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000'
  blankChrom = blankChrom[:(distinctLetters_cnt+1)]                                   #
  processingDataframe = df_in                                                         #
  for letter in nonDupFlatBoard :                                                     #
    processingDataframe = processingDataframe[processingDataframe != letter]          #
    chromStr = blankChrom[0:i]+'1'+blankChrom[i+1:]                                   #
    processingDataframe = processingDataframe.fillna(chromStr)                        #
    if letter == '' : ################################################################# until we get to this point. /\/\/\/\/\/\/\/\
      blankStr = chromStr ############################################################# we want to check each character for ' ' and store this for return
    i = i + 1                                                                         #
  chromFlat = processingDataframe.to_numpy().flatten()                                #
  chromBoard = ""                                                                     #
  for i2 in chromFlat :                                                               #
    chromBoard = chromBoard + i2                                                      #
  return blankStr ##################################################################### which we do here


def mkBoardChrom(df_in, chrom) : #as well as changing from board to chromesome, we also want to be able to do the inverse
  nonDupFlatBoard = dist_lett(df_in) #this code is the same as previously defined functions
  distinctLetters_cnt = dist_lett_cnt(df_in)
  step = distinctLetters_cnt + 1 # a "step", is how long each square's binary representation is, so number of letters + 1
  newBoard = []
  blankChrom = '000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000'
  blankChrom = blankChrom[:(distinctLetters_cnt+1)]
  a = df_in #until here. a is used in place of processingDataframe for ease of reading
  squareVal = [] #we want to init our variables
  xBoard = []
  for y in range(0,len(chrom),step): # we cycle through the chromesome and break up into the square values, number of letters + 1 - the step
    i = 0
    squareVal = chrom[y:y+step] #we store the binary representation of the square into a var
    newBoard.append(str(chrom[y:y+step])) #and then we append this
    for letter in nonDupFlatBoard : #we know the previously used order of the letters as per the dist_lett function which is consistant throughout the program
      chromStr = blankChrom[0:i]+'1'+blankChrom[i+1:] # when we identify the letter the square represnts
      if chromStr == squareVal :
        xBoard.append(str(letter)) #then we append this to a new board which is back to letters
      i = i + 1
  return xBoard

def scrabble_solve (df_in, word_list_path): #this funtion reads in the list of words and returns allowed four letter words
  words = pd.read_csv(word_list_path, sep=" ", header=None) #read in the text list using pandas
  words.columns = ["word"]
  words2 = words.to_numpy().flatten() #again, flatten the list for itteration
  i=0
  four_lett = [] #init
  for it in words2 :
    if len(str(it)) == 4 : #we go through all words in the dictionary
      four_lett.append(it) # if it has 4 letters we append, this is all 4 letter words
    i=i+1
  distinctLetters = dist_lett(df_in) #we need to know what letters the player has enetered
  distinctLetters_cnt = dist_lett_cnt(df_in)
  i=0
  four_lett_allow = four_lett #four_lett_allow will be our return, we want rid of four letter words that don't have our player's combination
  for it in distinctLetters[1:] : #go through each entered letter
    four_lett_allow = [ x for x in four_lett_allow if it  in x ] #use list comprehension to get the allowed words
  return four_lett_allow

def fitness(df_in, chrome_in, scrabble) : #our fitness function will run for a chromesome string, using an input df to represent the initial starting configuration of the board df_in
#the scrabble parameter will be supplied if solving to get a word at the bottom of the board
  fitness = 1 #we sart from a maximum fitness and will reduce the fitness as appropriate if rules are broken!
  distinctLetters_cnt = dist_lett_cnt(df_in) #again we need the number of distinct letters the player has entered
  chromBoard = mkChromBoard(df_in) #this vaiable changes the initial board entry into an equivelant chomesome string for coparison
  step = distinctLetters_cnt + 1 #this step is the length of the binary number representing the square
  chk = 0
  newBoard = [] #init
  blankStr = blankChr(df_in) #we will need to know how many blank squares are in the chromesome we are measuring, so we need to know how the program has represented it
  #loop incrementing step number (distinct letters plus 1)
  for y in range(0,len(chrome_in),step): #we want to chop up the chromesome into squares binary numbers again
    squareVal = chrome_in[y:y+step]
    newBoard.append(str(chrome_in[y:y+step])) #and then store them



    ##################################################################
    ##### rule 1: is the binary value for the square trying to set it to more than one letter (i.e 11100 is trying to say the square is "A AND B AND C" all at once)
    ##################################################################

    chk = int(squareVal.count('1')) #sum the values representing the square - it can only be one letter!
    if chk > 1 :
      fitness = fitness - 1 #if so, then reduce the fitness accordingly





  ##################################################################
  ##### rule 2: If row contains duplicate letters the fitness = fitness – 1
  ##################################################################

  newBoard = np.reshape(newBoard, (4,4)) #for this rule, we will use pandas so we need to reshape it back!
  newBoardDF = pd.DataFrame(newBoard) #before putting into df
  for i,j in newBoardDF.iterrows() : #j is the entire row and i will be the row number
    df_without_blanks = pd.DataFrame(np.delete(j,np.where(j == blankStr))) # we want a new dataframe without empty squares in it df_without_blanks
    dup = df_without_blanks.duplicated() #we then want to check for duplicate letters
    for k in dup :
      if k == True :
        fitness = fitness - 1






  ##################################################################
  ##### rule 3: If column contains duplicate letters the fitness = fitness – 1
  ##################################################################

  newBoardDFTrans = newBoardDF.transpose() #this logic will be identical to the above but we will just transpose the dataframe first to make the row check
  for i,j in newBoardDFTrans.iterrows() :  #actually be on the columns instead. logic is same as rule 2.
    l = pd.DataFrame(np.delete(j,np.where(j == blankStr)))
    dup = l.duplicated()
    for k in dup :
      if k == True :
        fitness = fitness - 1






  ##################################################################
  ##### rule 4: If Quadrant 1 contains duplicate letters the fitness = fitness – 1
  ##################################################################

  quadOne = newBoardDF.iloc[0:2,0:2].to_numpy().flatten() #we want to chop out the appropriate corner of the board from the datafarme
  quadOne = np.delete(quadOne,np.where(quadOne == blankStr)) #we then remove the empty squares again
  dupq = np.zeros(len(quadOne)) #we then want to apply a rule for identifying the uniquness of the characters within
  dupq[np.unique(quadOne, return_index=True)[1]] = True #Stackoverflow user ecatmur (2012) available at: https://stackoverflow.com/questions/11528078/determining-duplicate-values-in-an-array (accessed 29/09/2024)
  for k in dupq :
      if k == False :
        fitness = fitness - 0.25 #if the letter is not unique then we want to subtract one quarter (there are 4 squares per quadrant)







  ##################################################################
  ##### rule 5: If Quadrant 2 contains duplicate letters the fitness = fitness – 1
  ##################################################################

  quadTwo = newBoardDF.iloc[0:2,2:4].to_numpy().flatten() #we want to chop out the appropriate corner of the board from the datafarme
  quadTwo = np.delete(quadTwo,np.where(quadTwo == blankStr)) #we then remove the empty squares again
  dupq = np.zeros(len(quadTwo))
  dupq[np.unique(quadTwo, return_index=True)[1]] = True #Stackoverflow user ecatmur (2012) available at: https://stackoverflow.com/questions/11528078/determining-duplicate-values-in-an-array (accessed 29/09/2024)
  for k in dupq :
      if k == False :
        fitness = fitness - 0.25 #if the letter is not unique then we want to subtract one quarter (there are 4 squares per quadrant)







  ##################################################################
  ##### rule 6: If Quadrant 3 contains duplicate letters the fitness = fitness – 1
  ##################################################################

  quadThree = newBoardDF.iloc[2:4,0:2].to_numpy().flatten() #we want to chop out the appropriate corner of the board from the datafarme
  quadThree = np.delete(quadThree,np.where(quadThree == blankStr)) #we then remove the empty squares again
  dupq = np.zeros(len(quadThree))
  dupq[np.unique(quadThree, return_index=True)[1]] = True #Stackoverflow user ecatmur (2012) available at: https://stackoverflow.com/questions/11528078/determining-duplicate-values-in-an-array (accessed 29/09/2024)
  for k in dupq :
      if k == False :
        fitness = fitness - 0.25 #if the letter is not unique then we want to subtract one quarter (there are 4 squares per quadrant)







  ##################################################################
  ##### rule 7: If Quadrant 4 contains duplicate letters the fitness = fitness – 1
  ##################################################################

  quadFour = newBoardDF.iloc[2:4,2:4].to_numpy().flatten() #we want to chop out the appropriate corner of the board from the datafarme
  quadFour = np.delete(quadFour,np.where(quadFour == blankStr))
  dupq = np.zeros(len(quadFour))
  dupq[np.unique(quadFour, return_index=True)[1]] = True #Stackoverflow user ecatmur (2012) available at: https://stackoverflow.com/questions/11528078/determining-duplicate-values-in-an-array (accessed 29/09/2024)
  for k in dupq :
      if k == False :
        fitness = fitness - 0.25 #if the letter is not unique then we want to subtract one quarter (there are 4 squares per quadrant)







  ##################################################################
  ##### rule 8: board has to be filled out so subtract one sixteenth for each empty quare
  ##################################################################

  numberEmpty = 16 - (len(quadOne) + len(quadTwo) + len(quadThree) + len(quadFour))
  numberEmpty = 16 - (len(quadOne) + len(quadTwo) + len(quadThree) + len(quadFour))
  fitness = fitness - ((1/16) * numberEmpty)








  ##################################################################
  ##### rule 9: chromesomes cannot have squares that are no letter or empty, 00000
  ##################################################################

  #for chromesomes attempting 00000

  newBoard = np.reshape(newBoard, (4,4))
  newBoardDF = pd.DataFrame(newBoard)
  for i,j in newBoardDF.iterrows() :
    #index = [blankStr]
    l = pd.DataFrame(np.delete(j,np.where(j == '00000')))
    if len(j) != len(l) :
      fitness = fitness - 0.5
      #print('zeros')










  ##################################################################
  ##### rule 10: if we are processing as scrabble solution, then the last rowis any permitted word
  ##################################################################

  flatOrig = chromBoard.to_numpy().flatten() #for comparison, we want to get the original board flattened down
  flatNew = newBoardDF.to_numpy().flatten() # and same thing for new board
  brdLetts = mkBoardChrom(df_in,chrome_in) #for the scrabble solution, we want to get a one word representation of the final 4 squares of the board so we need the whole board as letters

  if len(scrabble) > 0 :

    #if scrabble condition is specified, we don't care if the original letters are lost. This is because it is very unlikely that a four letter word can be placed at the bottom of the board
    #if there are already letters above it in the board - then these will be duplicated straight away, breaking at least rule 3 above, but probably more



    try: #do this check in try catch un case the chomesome is malformed after slicing (because of invalid letters)
      last_row = brdLetts[len(brdLetts)-4:][0] + brdLetts[len(brdLetts)-4:][1] + brdLetts[len(brdLetts)-4:][2] + brdLetts[len(brdLetts)-4:][3]
      #we concatenate the final 4 letters of the board
    except :
      last_row = ''


    for it2 in scrabble : # and then compare to all allowed words
      if last_row == it2 :
        fitness = fitness + 9 # we add a massive fitness score here so that the fitness favours the chromesomes that have a word at bottom
                              # the rest of the prior checks all add to 1, as they would without scrabble check - so if we add 9 then we can divide the whole thinng by 10
                              # in this way, the values still ncrement, but we are only solving the REST of the board


    fitness = fitness / 10    # as per above, the potential fitness until this point for scrabble solutions is now up to 10, instead of 1 so divide by 10
                              # to keep original termination condition of fitness = 1 consistant irrespective of runtype

  else :

    ##################################################################
    ##### rule 11: if we are not runningas scrablle solution, the the original placement of the letters should be retained
    ##################################################################

       # if it isn't scrabble, then the original board layout should be retained. We do this by comparing every square in new to every square in old and reducing the fitness
       # where any non blank initial value has changed
    i = 0
    for  it1 in flatOrig :
      if (it1 != flatNew[i]) and (it1 != blankStr ) :
       fitness = fitness - 1
      i = i+1

  return fitness

#this function generates the initial population of random chromesomes
def initial_pop(df_initial, popn, scrabble):
    fitPop = []
    chrom = mkChrom(df_initial)
    pop = []  # this will be our returned population from the function

    ID = []
    lc = dist_lett_cnt(df_initial)
    dL = dist_lett(df_initial)
    siz = (lc + 1) * 16

    if len(scrabble) == 0:
        pop.append(chrom)  # include the initial board
    # j=1
    # ID.append(j)

    if len(scrabble) > 0:  # if we are running the program for scrabble solution, then we want to put empty boards WITH the valid words at the bottom into the initial population
        for slet in scrabble:  # this will ensure that the program is starting from the word and working out what the other squares should be to achieve this
            # j=j+1
            dft = []
            dft = pd.DataFrame(
                [['', '', '', ''], ['', '', '', ''], ['', '', '', ''], [slet[0], slet[1], slet[2], slet[3]]])

            # we need to encode the letters in the scrabble board in correct order relative to the order in the initial board
            blankChrom = '00000'
            i = 0
            for letter in dL:
                dft = dft[dft != letter]  # replace the letter in the copied processingDataframe with null
                chromStr = blankChrom[0:i] + '1' + blankChrom[
                                                   i + 1:]  # the string representing whether the square is the letter is 1 for the position through the blankChom. This is NOT the binary number representing the itteration through the letters, i
                dft = dft.fillna(chromStr)  # replace the blanks with the string representing the letter
                print(dft)
                i = i + 1
            cT = ""
            flatSBoard = []
            flatSBoard = dft.to_numpy().flatten()
            t = ""
            for t in flatSBoard:
                cT = cT + str(t)
            print(cT)
            pop.append(cT)

    candid = ""  # this is init of string that will be candidate for solving the board (chromesome)
    init_size = len(chrom)  # we get a measure of the length required for a chromesome
    for i in range(len(ID), popn - len(
            ID) - 1):  # we pass in the parameter popn which disctates how many chromesomes are in our initial population
        num = random.getrandbits(init_size + 10)  # we generate a random number
        candid = str(format(num, '0b'))[:siz]  # before making it 0's and 1's and chopping at required size
        pop.append(candid)  # we then add it to the initial population

    i = 0
    for cp in pop:  # we then want to add a column for the fitness of each initial chromesome
        fitPop.append(fitness(df_initial, cp, scrabble))
        ID.append(i)  # we need to set an ID for the chromesome
        i = i + 1

    dict = {'chromPop': pop, 'fitness': fitPop, 'ID': ID}  # we need to define the combined df
    df = pd.DataFrame(dict)

    df = df.sort_values(by=['fitness'], ascending=False).reset_index(drop=True)  # and then sort it in decending fitness
    # log_df.append(df)
    return df

#this function defines the crossover and mutation
def crossover_mutate(df_in, population,cp,pm, mut) : # we pass in the followig parameters to control the mutation and crossover:
  i=0  #we start at position 0                       # 1. df_in: this is the dataframe for the initial board
                                                     # 2. population: this is the population requred after crossover and mutation
                                                     # 3. cp: this is the crossover rate.
                                                     #        we have decided to set the crossover point randomly but retain cp as
                                                     #        a control over the amount of parents and children to keep
                                                     # 4. pm: this is the mutation rate
                                                     # 5. mut: this variable determines the complexity of mutation by mutating the children multiple times if set
  new_pop = []
  new_ID = []
  log_dfr = []
  distinctLetters_cnt = dist_lett_cnt(df_in)


  #######################################
  # Crossover
  #######################################



  while len(new_pop) < population : # we want to do this until the population is the right size!
    df2 = df_in.iloc[i:i+2] # we take the next two chromesomes
    df2 = df_in["chromPop"] # and we only care about their binary representation
    df_id = df_in["ID"] # and their ID
    #print(df_id)
    parent1 = df2.iloc[0] # the first parent is the first chromesome
    p1_id = df_id.iloc[0] # and the ID for the first parent
    parent2 = df2.iloc[1] # and the second the next
    p2_id = df_id.iloc[1] # and the ID for the second parent


    # we want to set a random crossover point, but keep it at a square value otherwise we
    # lose the advantage of combining parents (as we might split a chomesome in the middle of a square)
    lc = dist_lett_cnt(df_in) # lc = letter count
    siz = (lc + 1) * 16   # siz is the size of a chromesome
    crossover_point = random.randint(0,(siz)//2)  # we get some random int that is less than or equal to halfway along a chomesome
    crossover_point = crossover_point / (lc + 1) # we divide by the (lc + 1) which is the size of the chomesome for a square (usually 5 for a 4x4 board)
    crossover_point = round(crossover_point,0)   # we round this number to zero decimal places to eliminate the remainder
    crossover_point = int(crossover_point * (lc + 1)) # we then multiply back up to get nearest (lc + 1)

    child1 = parent1[0:crossover_point] + parent2[crossover_point:siz] # we splice the parents at the crossover point to make the children
    child2 = parent2[0:crossover_point] + parent1[crossover_point:siz]

    #######################################
    # mutate the children
    #######################################

    # we now need to mutate the children.
    # This section is written for specifically 4 letters and this is a known limitation of the program
    # future work would be required to cater for longer chromesomes for the square by making the list
    # of square letter binary strings dynamic as it is elsewhere in the code

    if random.random() > 1-pm : # based on the mutation rate pm
      for i in range(0,mut) : # how many mutations to make is mut variable, this lets us introduce multiple mutations
        pos_mut = random.randint(0,len(df2.iloc[0]) - 5) # we need to choose a random location to mutate
        pos_mut = pos_mut / 5                            # divide by 5 (this should be dynamic)
        pos_mut = round(pos_mut,0)                       # round to get rid of decimal remainder
        pos_mut = int(pos_mut * 5)                       # get back to the nearest 5(the nearest start to a letter chosen at random)

        child1 = child1[0:pos_mut] + random.choice(['01000','00100','00010','00001']) + child1[pos_mut+5:80] # inject a new letter at randm at a random place

        child2 = child2[0:pos_mut] + random.choice(['01000','00100','00010','00001']) + child2[pos_mut+5:80] # into both children

      # get rid of "zero" entries
      # there can be a case where the gene for the square is zero for everything meaning that the gene for that square
      # is actually telling it to do nothing, which is neither valid nor helpful
      for y in range(0,len(child1),5): #for each square
        letr = child1[y:y+5] # of the first child
        if letr == '00000': #get rid of all zero entries and replace with radnomly chosen letter
          child1 = child1[0:y] + random.choice(['10000','01000','00100','00010','00001']) + child1[y+5:len(child1)]

        # we can add in another mutation at this point for better performance, by additionally (and based on pm)
        # switching out blank values with random letter!

        if letr == '10000' and random.random() > 1-pm:
          child1 = child1[0:y] + random.choice(['10000','01000','00100','00010','00001']) + child1[y+5:len(child1)]

      for y in range(0,len(child2),5): #repeat the above for child 2
        letr = child2[y:y+5]
        if letr == '00000':
          child2 = child2[0:y] + random.choice(['10000','01000','00100','00010','00001']) + child2[y+5:len(child2)]
        if letr == '10000' and random.random() > 1-pm:
          child2 = child2[0:y] + random.choice(['10000','01000','00100','00010','00001']) + child2[y+5:len(child2)]



    #######################################
    # new population
    #######################################

    # we now want to begin making our new population
    if i < int(round(population/20)) : # the incoming population is sorted baseed on fitness, so we keep top 20 % of parents as well - strongest survive
      new_pop.append(parent1)
      new_pop.append(parent2)
      new_pop.append( child1)
      new_pop.append( child2)
    else :
      if float(random.random()) > cp : # for the remaining, we use a random value set aginst the cp (crossover rate, not point) to determine whether we keep the children we have made
        new_pop.append(parent1)
        new_pop.append(parent2)
        new_pop.append( child1)
        new_pop.append( child2)
      else :
        new_pop.append( child1) # else we just keep the children to give them a chance as parents have lower score than to 20%
        new_pop.append( child2)
    log_dfr.append(new_pop)
    i = i + 2
  return new_pop, log_dfr

def GA_Lop(output_box, input_grid, iPop, population,cp,pm,mut,scrabble,maxl) :   #the inputs here are as follows:
  output_box.insert(tk.END, "GA Lop algorithm executed.\n")
  df_log = []
  nit = 0                                                #                                1. iPop: initial population
  new_iPop = []                                          #                                2. population: population size
  new_iPop, log_dfr = crossover_mutate(iPop, population,cp,pm,mut)#                       3. cp: this is the crossover rate
  fitNewPop = []
  new_id = []                                            #                                4. pm: mutation rate
  p = 0                                                  #                                5. mut: mutation rate
  dict = []                                              #                                6. scrabble: allowed list of words.empty denotes non-scrabble run
  for x in new_iPop[0:] :   #get rid                         #                                7. maxl: "max loop" maximum number of generations to try for
    if len(x) == 80 :   #of invalid chromesomes for safety
      fitNewPop.append(fitness(input_grid, x, scrabble)) # for correctly formed, we get the fitnesss value for new pop
      new_id.append(p)
    else :
      new_iPop.pop(p)
    p = p+1

  # we are going to make a new combined dataframe for the new generation containing the fitness
  dict = {'chromPop': new_iPop, 'fitness': fitNewPop, 'ID':new_id}
  df1 = pd.DataFrame(dict)
  df1 = df1.sort_values(by=['fitness'], ascending=False).reset_index(drop=True) # sorted by fitness
  df_log = df1

  max_gen_fit = 0 #init
  max_gen_fit = df1.fitness.values[0] #get the first value of fitness, it is sorted descending
  nit = nit + 1
  print('gen',nit,' fitness: ', max_gen_fit) #display to user
  output_box.insert(tk.END, f"gen',{nit},' fitness: ', {max_gen_fit}\n")
  print('gen', nit,' board corresponding to fittest chromesome ',df1.loc[0].chromPop) #also show chromesome
  output_box.insert(tk.END, f"gen',{nit},' board corresponding to fittest chromesome: ', {df1.loc[0].chromPop}\n")
  print('gen', nit, ' corresponding to initial chrome ID: ', df1.loc[0].ID)  # alongside the chrome
  try: #wrap the next bit in try catch in case the board has invalid letters in it
    print(np.reshape(mkBoardChrom(input_grid,df1.loc[0].chromPop),(4,4))) # show the reconstructed board to the user
    output_box.insert(tk.END, f"board:',{np.reshape(mkBoardChrom(input_grid, df1.loc[0].chromPop), (4, 4))}\n")
  except :
    print('chromesome not a board')




  term_cond = 1 # this is our termination condition for the GA, fitness = 1

  l=0 #init
  while max_gen_fit < term_cond and l < maxl: #keep repeating this if the termination condition isn't satisifed and we haven't reached maximim number of generations to try for
    new_iPop = []
    log_dfr2 = []
    new_iPop, log_dfr2  = crossover_mutate(df1, population,cp,pm,mut) #run crossover andmutate
    log_dfr = log_dfr + log_dfr2
    dict = []
    #print(new_iPop)
    fitNewPop = []
    new_id = []
    p = 0
    for x in new_iPop[0:] :
      if len(x) == 80 :
        fitNewPop.append(fitness(input_grid, x, scrabble)) #add the fitness to the new population
        new_id.append(p)
      else :
        new_iPop.pop(p)
      p = p+1


    if len(new_iPop) == len(fitNewPop) :  # added during triage of issue where new population fitness function failed for one record so the dataframe couldn't be combined
      df1 = []                            # we have retained because they should alsways match
      dict = {'chromPop': new_iPop, 'fitness': fitNewPop, 'ID':new_id}
      df1 = pd.DataFrame(dict)            # combine the dataframes into new
      df1 = df1.sort_values(by=['fitness'], ascending=False).reset_index(drop=True) # and then sort desceding by fitness



      max_gen_fit = df1.fitness.values[0] #get the first value of fitness, it is sorted descending
      nit = nit + 1
      iID = df1.loc[0].ID
      print('gen',nit,' fitness: ', max_gen_fit) # display fitness for the gen to the user
      output_box.insert(tk.END, f"gen',{nit},' fitness: ', {max_gen_fit}\n")
      print('gen', nit,' board corresponding to fittest chromesome ',df1.loc[0].chromPop) # alongside the chrome
      output_box.insert(tk.END, f"gen',{nit},' board corresponding to fittest chromesome: ', {df1.loc[0].chromPop}\n")
      df_log = pd.concat([df_log, df1], ignore_index=True)
      # print(df_log)
      print('gen', nit, ' corresponding to initial chrome ID: ', df1.loc[0].ID)
      try: #wrap the next bit in try catch in case the board has invalid letters in it
        print(np.reshape(mkBoardChrom(input_grid,df1.loc[0].chromPop),(4,4)))  # show the reconstructed board to the user
        output_box.insert(tk.END, f"board:',{np.reshape(mkBoardChrom(input_grid,df1.loc[0].chromPop),(4,4))}\n")
      except :
        print('chromesome not a board')
    l = l + 1

  if l == maxl : # if the solution hasn't been found then tell the user
    print('solution not found in ',l,' generations')
  else :
    print('solution found in ',nit) # otherwise tell the user
    output_box.insert(tk.END, f"solution found in',{nit}\n")
  return df1,nit, iID, df_log,l # this is df containing solution (or best alternative) and number of generations it took


# Function to validate user input
def validate_input(input_grid):
    # Collect all distinct letters entered by the user
    distinct_letters = set()
    for row in input_grid:
        for letter in row:
            if letter and letter != ' ':  # Ignore empty entries
                distinct_letters.add(letter)

    # Check if there are exactly four distinct letters
    if len(distinct_letters) != 4:
        return False, "There must be only four unique letters in the grid."

    # Check rows for unique letters
    for row in input_grid:
        unique_letters = set()  # Set to track unique letters
        for letter in row:
            if letter and letter != ' ':  # Ignore empty entries
                if letter in unique_letters:
                    return False, "The same letter cannot appear more than once in the same row."
                unique_letters.add(letter)

    # Check columns for unique letters
    for i in range(4):
        unique_letters = set()  # Set to track unique letters
        for j in range(4):
            letter = input_grid[j][i]  # Get the letter in the current column
            if letter and letter != ' ':  # Ignore empty entries
                if letter in unique_letters:
                    return False, "The same letter cannot appear more than once in the same column."
                unique_letters.add(letter)

    # Check 2x2 subgrids for unique letters
    for x in range(0, 4, 2):  # Iterate over starting rows of subgrids
        for y in range(0, 4, 2):  # Iterate over starting columns of subgrids
            unique_letters = set()  # Set to track unique letters
            for i in range(2):  # Iterate over rows in the 2x2 subgrid
                for j in range(2):  # Iterate over columns in the 2x2 subgrid
                    letter = input_grid[x + i][y + j]  # Get the letter in the subgrid
                    if letter and letter != ' ':  # Ignore empty entries
                        if letter in unique_letters:
                            return False, "The same letter cannot appear more than once in the same 2x2 subgrid."
                        unique_letters.add(letter)

    return True, ""  # If all checks pass, return True


# Class to create the Sudoku Solver GUI
class SudokuSolverApp:
    def __init__(self, master):
        self.master = master  # Save the reference to the main window
        master.title("Soduku!")  # Set the title of the window

        self.grid_entries = [[None for _ in range(4)] for _ in range(4)]  # Create a 4x4 grid for entries
        self.create_grid()  # Call method to create the grid

        self.algorithm_var = tk.StringVar(value="genetic")  # Default to genetic algorithm

        # Create and place a label for algorithm selection
        self.algorithm_label = tk.Label(master, text="Select Algorithm:")
        self.algorithm_label.grid(row=4, column=0, columnspan=4, sticky='w')  # Left align

        # Create and place radio buttons for algorithm selection
        self.genetic_radio = tk.Radiobutton(master, text="Real-valued Algorithm", variable=self.algorithm_var, value="genetic")
        self.genetic_radio.grid(row=5, column=0, columnspan=4, sticky='w')  # Left align

        self.placeholder_radio = tk.Radiobutton(master, text="Binary Algorithm", variable=self.algorithm_var, value="placeholder")
        self.placeholder_radio.grid(row=6, column=0, columnspan=4, sticky='w')  # Left align

        self.scrabble_radio = tk.Radiobutton(master, text="Scrabble Solve", variable=self.algorithm_var,
                                             value="scrabble")
        self.scrabble_radio.grid(row=7, column=0, columnspan=4, sticky='w')  # Left align

        # Create and place a button to solve the puzzle
        self.solve_button = tk.Button(master, text="Solve", command=self.solve)
        self.solve_button.grid(row=8, column=0, columnspan=4, sticky='w')  # Left align

        # Create and place a text box for output
        self.output_box = scrolledtext.ScrolledText(root, width=50, height=20, fg="white")
        self.output_box.grid(row=9, column=0, columnspan=4, sticky='w')

        # Configure grid to avoid gaps
        for col in range(4):
            master.grid_columnconfigure(col, weight=1)  # Allow columns to expand evenly

    # Method to create the 4x4 grid of entry boxes
    def create_grid(self):
        for i in range(4):  # Iterate over rows
            for j in range(4):  # Iterate over columns
                entry = tk.Entry(self.master, width=5, font=('Arial', 18), justify='center')  # Create entry box
                entry.grid(row=i, column=j, padx=0, pady=0, sticky='w')  # Remove extra space between boxes and left align
                self.grid_entries[i][j] = entry  # Save the reference to the entry

    # Method to get the current input from the grid
    def get_input_grid(self):
        input_grid = []  # Initialize an empty list to store the input grid
        for row in self.grid_entries:  # Iterate over each row
            input_row = []  # Initialize an empty list for the current row
            for entry in row:  # Iterate over each entry in the row
                value = entry.get().strip().upper()  # Get the value, strip whitespace, and convert to uppercase
                input_row.append(value if value else '')  # Append value or empty string for empty cells
            input_grid.append(input_row)  # Append the input row to the input grid
        print(input_grid)
        return input_grid  # Return the complete input grid

    # Method to display the solution in the grid
    def display_solution(self, solution):
        for i in range(4):  # Iterate over rows
            for j in range(4):  # Iterate over columns
                self.grid_entries[i][j].delete(0, tk.END)  # Clear the current entry
                self.grid_entries[i][j].insert(0, solution[i][j])  # Insert the solution value

    # Method to solve the puzzle
    def solve(self):
        input_grid = self.get_input_grid()  # Get the input grid from entries
        input_df = pd.DataFrame(input_grid)
        valid, message = validate_input(input_grid)  # Validate the input
        if not valid:  # If the input is invalid
            messagebox.showerror("Input Error", message)  # Display error message
            return

        self.output_box.delete(1.0, tk.END)  # Clear previous output

        # Determine which algorithm to use
        selected_algorithm = self.algorithm_var.get()
        if selected_algorithm == "genetic":
            solution = genetic_algorithm(input_grid, self.output_box)  # Solve using genetic algorithm
        elif selected_algorithm == "placeholder":

            ########################################
            # define parameters of the GA
            ########################################

            cp = 0.9  # this corresponds to 90% chance of crossover retention. in literatre cp is crossoverpoint, so probably should have changed this!
            pm = 0.4  # this is the mutation rate, set to 40% through trial and error
            population = 500  # a population of 500 has been selected to balance performance and solve time thrugh trial and error
            mut_num = 2  # this is the complexity of mutation as defined in the function. 2 seems to balance well for this GA
            maximum_generations = 1000  # this is set high, to 1000 but generally solves under 100

            ########################################
            # set Scrabble parameters
            ########################################

            # scrab_list = scrabble_solve (board, 'scrabble_words.txt') # if running and trying to find a word, run the scrabble_solve function to get the allowed words
            # print('program trying to solve for word along the bottom. One of: ',scrab_list) # show user the target words

            scrab_list = []  # IF RUNNING WITHOUT TRYNG TO FIND A WORD THEN UNCOMMENT THIS LINE TO CLEAR THE LIST

            ########################################
            # run the GA
            ########################################
            log_df = []
            whole_log = []
            entire = []
            gen = 0

            iPop = initial_pop(input_df, population, scrab_list)  # Generate initial population
            # solution_df, _ = GA_Lop(self.output_box, input_df, iPop, population, cp, pm, mut_num, scrab_list, maximum_generations)  # Run GA_Lop
            solution_df, nit, iID, whole_log, gen = GA_Lop(self.output_box, input_df, iPop, population, cp, pm, mut_num, scrab_list, maximum_generations)
            solution_steps = whole_log[whole_log['ID'] == iID]
            print(' ')
            print('Generation 0 (initial population)')
            iPop_ip = iPop[iPop['ID'] == iID]["chromPop"].values[0]
            # print(iPop_ip)
            # print(np.reshape(mkBoardChrom(board,iPop_ip),(4,4)))

            try:  # wrap the next bit in try catch in case the board has invalid letters in it
                print(iPop_ip)
                print(np.reshape(mkBoardChrom(input_df, iPop_ip), (4, 4)))  # show the reconstructed board to the user
            except:
                print(iPop_ip)
                print('chromesome not a board')

            i = 1
            for it in solution_steps["chromPop"]:
                try:
                    print(' ')
                    print('Generation ', i)
                    print(it)
                    print(np.reshape(mkBoardChrom(input_df, it), (4, 4)))
                except:
                    print(' ')
                    print('Generation ', i)
                    print(it)
                    print('not a board')
                i = i + 1

            # Convert DataFrame to grid if needed
            print("Solution DataFrame:", solution_df)  # Check the structure
            solution_grid = mkBoardChrom(input_df, solution_df['chromPop'].iloc[0])  # Convert chromosome to flat list
            solution =  [solution_grid[i:i + 4] for i in range(0, len(solution_grid), 4)]  # Convert chromosome to flat list
            print("Solution :", solution)  # Check the structure
        elif selected_algorithm == "scrabble":
            ########################################
            # define parameters of the GA
            ########################################

            cp = 0.9  # this corresponds to 90% chance of crossover retention. in literatre cp is crossoverpoint, so probably should have changed this!
            pm = 0.4  # this is the mutation rate, set to 40% through trial and error
            population = 500  # a population of 500 has been selected to balance performance and solve time thrugh trial and error
            mut_num = 2  # this is the complexity of mutation as defined in the function. 2 seems to balance well for this GA
            maximum_generations = 1000  # this is set high, to 1000 but generally solves under 100

            ########################################
            # set Scrabble parameters
            ########################################

            scrab_list = scrabble_solve (input_df, 'scrabble_words.txt') # if running and trying to find a word, run the scrabble_solve function to get the allowed words
            print('program trying to solve for word along the bottom. One of: ',scrab_list) # show user the target words

            scrab_list = []  # IF RUNNING WITHOUT TRYNG TO FIND A WORD THEN UNCOMMENT THIS LINE TO CLEAR THE LIST

            ########################################
            # run the GA
            ########################################
            log_df = []
            whole_log = []
            entire = []
            gen = 0

            iPop = initial_pop(input_df, population, scrab_list)  # Generate initial population
            # solution_df, _ = GA_Lop(self.output_box, input_df, iPop, population, cp, pm, mut_num, scrab_list, maximum_generations)  # Run GA_Lop
            solution_df, nit, iID, whole_log, gen = GA_Lop(self.output_box, input_df, iPop, population, cp, pm, mut_num,
                                                           scrab_list, maximum_generations)
            solution_steps = whole_log[whole_log['ID'] == iID]
            print(' ')
            print('Generation 0 (initial population)')
            iPop_ip = iPop[iPop['ID'] == iID]["chromPop"].values[0]
            # print(iPop_ip)
            # print(np.reshape(mkBoardChrom(board,iPop_ip),(4,4)))

            try:  # wrap the next bit in try catch in case the board has invalid letters in it
                print(iPop_ip)
                print(np.reshape(mkBoardChrom(input_df, iPop_ip), (4, 4)))  # show the reconstructed board to the user
            except:
                print(iPop_ip)
                print('chromesome not a board')

            i = 1
            for it in solution_steps["chromPop"]:
                try:
                    print(' ')
                    print('Generation ', i)
                    print(it)
                    print(np.reshape(mkBoardChrom(input_df, it), (4, 4)))
                except:
                    print(' ')
                    print('Generation ', i)
                    print(it)
                    print('not a board')
                i = i + 1


            # Convert DataFrame to grid if needed
            print("Solution DataFrame:", solution_df)  # Check the structure
            solution_grid = mkBoardChrom(input_df, solution_df['chromPop'].iloc[0])  # Convert chromosome to flat list
            solution = [solution_grid[i:i + 4] for i in
                        range(0, len(solution_grid), 4)]  # Convert chromosome to flat list
            print("Solution :", solution)  # Check the structure

        if solution:  # If a solution was found
            self.display_solution(solution)  # Display the solution in the grid

# Entry point of the program
# Entry point of the program
if __name__ == "__main__":
    root = tk.Tk()  # Create the main window
    app = SudokuSolverApp(root)  # Create an instance of the SudokuSolverApp

    root.mainloop()  # Run the GUI event loop
