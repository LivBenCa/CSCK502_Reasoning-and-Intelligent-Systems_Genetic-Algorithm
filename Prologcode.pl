{\rtf1\ansi\ansicpg1252\cocoartf2761
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 :- use_module(library(random)).\
:- use_module(library(lists)).\
\
% Define the grid, letters, and constraints\
grid(4, 4).\
letters([w, o, r, d]).\
\
% Generate a random grid that satisfies the initial configuration\
random_grid(N, M, Grid) :-\
    length(Grid, N),\
    maplist(random_row(M), Grid).\
\
random_row(M, Row) :-\
    length(Row, M),\
    letters(Letters),\
    maplist(random_member_from(Letters), Row).\
\
random_member_from(List, Elem) :-\
    random_member(Elem, List).\
\
% Fill the grid based on the initial configuration (this is a dummy function for now)\
fill_grid([], _).\
fill_grid([Row|Rest], Letter) :-\
    maplist(fill_cell(Letter), Row),\
    fill_grid(Rest, Letter).\
\
fill_cell(Letter, Cell) :-\
    (Cell == '' -> random_member(Letter, letters) ; true).\
\
% Check if the grid is valid (all constraints are met)\
valid_grid(Grid) :-\
    rows_valid(Grid),\
    columns_valid(Grid),\
    subgrids_valid(Grid).\
\
rows_valid(Grid) :-\
    forall(member(Row, Grid), unique_elements(Row)).\
\
columns_valid(Grid) :-\
    transpose(Grid, Transposed),\
    rows_valid(Transposed).\
\
subgrids_valid(Grid) :-\
    subgrids(Grid, Subgrids),\
    forall(member(Subgrid, Subgrids), unique_elements(Subgrid)).\
\
unique_elements(List) :-\
    sort(List, Sorted),\
    letters(Letters),\
    permutation(Letters, Sorted).\
\
subgrids(Grid, Subgrids) :-\
    findall(Subgrid, subgrid(Grid, Subgrid), Subgrids).\
\
subgrid(Grid, Subgrid) :-\
    grid(4, 4),\
    Subgrid = [\
        Grid[0][0], Grid[0][1], Grid[1][0], Grid[1][1],\
        Grid[0][2], Grid[0][3], Grid[1][2], Grid[1][3],\
        Grid[2][0], Grid[2][1], Grid[3][0], Grid[3][1],\
        Grid[2][2], Grid[2][3], Grid[3][2], Grid[3][3]\
    ].\
\
% Check if a specific word is present on the edges of the grid\
word_present(Grid, Word) :-\
    edges(Grid, Edges),\
    member(Word, Edges).\
\
edges(Grid, Edges) :-\
    nth0(0, Grid, Top),\
    nth0(3, Grid, Bottom),\
    maplist(nth0(0), Grid, LeftCol),\
    maplist(nth0(3), Grid, RightCol),\
    atomic_list_concat(Top, TopEdge),\
    atomic_list_concat(Bottom, BottomEdge),\
    atomic_list_concat(LeftCol, LeftEdge),\
    atomic_list_concat(RightCol, RightEdge),\
    Edges = [TopEdge, BottomEdge, LeftEdge, RightEdge].\
\
% Calculate the fitness of a grid\
fitness(Grid, Word, FitnessScore) :-\
    valid_grid(Grid),\
    (word_present(Grid, Word) -> FitnessScore is 100 ; FitnessScore is 0).\
\
% Genetic algorithm\
genetic_algorithm(MaxGenerations, PopulationSize, CrossoverRate, MutationRate, Solution) :-\
    generate_initial_population(PopulationSize, Population),\
    genetic_algorithm_loop(Population, MaxGenerations, CrossoverRate, MutationRate, Solution).\
\
genetic_algorithm_loop(Population, 0, _, _, Solution) :-\
    best_solution(Population, Solution).\
genetic_algorithm_loop(Population, MaxGenerations, CrossoverRate, MutationRate, Solution) :-\
    evaluate_population(Population, Fitnesses),\
    select_parents(Population, Fitnesses, Parents),\
    crossover_population(Parents, CrossoverRate, Offspring),\
    mutate_population(Offspring, MutationRate),\
    append(Population, Offspring, NewPopulation),\
    sort(2, @>=, NewPopulation, SortedPopulation), % Sort based on fitness\
    length(SortedPopulation, L),\
    length(Population, PopSize),\
    length(NewPopulation1, PopSize),\
    append(NewPopulation1, _, SortedPopulation),\
    NewMaxGenerations is MaxGenerations - 1,\
    genetic_algorithm_loop(NewPopulation1, NewMaxGenerations, CrossoverRate, MutationRate, Solution).\
\
% Generate initial population\
generate_initial_population(PopulationSize, Population) :-\
    length(Population, PopulationSize),\
    maplist(random_grid(4, 4), Population).\
\
% Evaluate population fitness\
evaluate_population(Population, Fitnesses) :-\
    maplist(fitness_with_word(word), Population, Fitnesses).\
\
fitness_with_word(Word, Grid, Fitness) :-\
    fitness(Grid, Word, Fitness).\
\
% Select parents using tournament selection\
select_parents(Population, Fitnesses, Parents) :-\
    length(Population, Len),\
    NumParents is Len // 2,\
    findall(Parent, (between(1, NumParents, _), tournament_selection(Population, Fitnesses, Parent)), Parents).\
\
tournament_selection(Population, Fitnesses, Winner) :-\
    random_select(A, Population, _),\
    random_select(B, Population, _),\
    nth0(IndexA, Population, A),\
    nth0(IndexB, Population, B),\
    nth0(IndexA, Fitnesses, FitnessA),\
    nth0(IndexB, Fitnesses, FitnessB),\
    (FitnessA > FitnessB -> Winner = A ; Winner = B).\
\
% Crossover population\
crossover_population([], _, []).\
crossover_population([Parent1, Parent2 | Rest], CrossoverRate, [Child1, Child2 | Offspring]) :-\
    (random_float < CrossoverRate ->\
        one_point_crossover(Parent1, Parent2, Child1, Child2)\
        ;\
        Child1 = Parent1, Child2 = Parent2),\
    crossover_population(Rest, CrossoverRate, Offspring).\
\
one_point_crossover(Parent1, Parent2, Child1, Child2) :-\
    random_between(1, 15, Point),\
    append(Head1, Tail1, Parent1),\
    append(Head2, Tail2, Parent2),\
    length(Head1, Point),\
    length(Head2, Point),\
    append(Head1, Tail2, Child1),\
    append(Head2, Tail1, Child2).\
\
% Mutate population\
mutate_population([], _).\
mutate_population([Grid | Rest], MutationRate) :-\
    mutate_grid(Grid, MutationRate),\
    mutate_population(Rest, MutationRate).\
\
mutate_grid(Grid, MutationRate) :-\
    maplist(mutate_row(MutationRate), Grid).\
\
mutate_row(MutationRate, Row) :-\
    maplist(mutate_cell(MutationRate), Row).\
\
mutate_cell(MutationRate, Cell) :-\
    (random_float < MutationRate ->\
        letters(Letters),\
        random_member(NewCell, Letters),\
        Cell = NewCell\
        ;\
        true).\
\
% Check if a solution is found\
solution_found(Grid) :-\
    fitness(Grid, word, 100).\
\
% Find the best solution\
best_solution(Population, BestSolution) :-\
    maplist(fitness_with_word(word), Population, Fitnesses),\
    max_member(BestFitness, Fitnesses),\
    nth0(Index, Fitnesses, BestFitness),\
    nth0(Index, Population, BestSolution).\
\
% Display solution\
display_solution(Grid) :-\
    maplist(writeln, Grid),\
    nl.\
\
% Test the genetic algorithm\
test_genetic_algorithm :-\
    MaxGenerations = 1000,\
    PopulationSize = 100,\
    CrossoverRate = 0.7,\
    MutationRate = 0.01,\
    genetic_algorithm(MaxGenerations, PopulationSize, CrossoverRate, MutationRate, Solution),\
    display_solution(Solution).\
\
% Run the test\
:- test_genetic_algorithm.\
}