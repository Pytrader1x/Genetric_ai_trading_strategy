import numpy as np
import matplotlib.pyplot as plt

def cal_pop_fitness(equation_inputs, pop, opt=0, penalty_mode=False, penalty_settings=None):
    """
    Calculates the fitness (SSR) for each candidate weight vector.
    
    Parameters:
      - equation_inputs: NumPy array; first column is log returns (logr), the rest are trading rule signals.
      - pop: Population of candidate weight vectors (shape: [num_candidates, num_weights]).
      - opt: (unused) selection mode.
      - penalty_mode: Boolean flag; if True, applies additional penalties.
      - penalty_settings: Dictionary specifying penalty settings for:
           'hold'  : { 'active': bool, 'target': int, 'beta': float }
           'win'   : { 'active': bool, 'target': float, 'beta': float }
           'loss'  : { 'active': bool, 'target': float, 'beta': float }
         For example:
         {
           'hold': {'active': True, 'target': 10, 'beta': 0.5},
           'win' : {'active': True, 'target': 2,  'beta': 0.5},
           'loss': {'active': True, 'target': 1,  'beta': 0.5}
         }
         The idea is: if a candidate's average holding period is below target,
         or its average pip gain (for wins) is below target, or average pip loss (for losses)
         is below target, then the candidate's raw SSR is divided by a penalty factor.
    
    Returns:
      - adjusted_SSR: NumPy array of fitness values (higher is better).
    """
    # Extract log returns and compute positions for each candidate.
    logr = equation_inputs[:, 0]  # shape: (n_timesteps,)
    positions = pop @ equation_inputs[:, 1:].T  # shape: (num_candidates, n_timesteps)
    port_r = (positions * logr).astype(np.float64)
    
    # Compute raw SSR for each candidate:
    # Note: The negative sum in denominator is to penalize negative returns.
    raw_SSR = np.mean(port_r, axis=1) / np.std(port_r, axis=1) / (-np.sum(port_r[port_r < 0]))
    
    # If no penalty mode, simply return raw_SSR.
    if not penalty_mode or penalty_settings is None:
        return raw_SSR
    
    # Use default settings if not provided:
    defaults = {
        'hold': {'active': True, 'target': 10, 'beta': 0.5},
        'win':  {'active': True, 'target': 2,  'beta': 0.5},
        'loss': {'active': True, 'target': 1,  'beta': 0.5}
    }
    # Merge user-provided settings with defaults:
    for key in defaults:
        if key not in penalty_settings:
            penalty_settings[key] = defaults[key]
        else:
            for subkey in defaults[key]:
                if subkey not in penalty_settings[key]:
                    penalty_settings[key][subkey] = defaults[key][subkey]
    
    # Compute a normalized price series from logr (assume base price = 1)
    # This is used to simulate trades and compute pip differences.
    # Make sure logr is properly processed as a numpy array
    logr_array = np.asarray(logr).flatten()
    # We don't actually need price_series for the penalties calculation
    # Just set a base pip size without creating the full series
    pip_size = 0.0001  # 1 pip = 0.0001
    
    penalties = []
    n_timesteps = positions.shape[1]
    
    for i in range(positions.shape[0]):
        pos = positions[i, :]
        
        # --- Calculate average holding period ---
        sign_changes = np.sum(np.abs(np.diff(np.sign(pos))))
        num_trades = sign_changes / 2 if sign_changes > 0 else 0
        avg_hold = n_timesteps if num_trades == 0 else n_timesteps / num_trades
        
        if penalty_settings['hold']['active'] and avg_hold < penalty_settings['hold']['target']:
            penalty_hold = 1 + penalty_settings['hold']['beta'] * (penalty_settings['hold']['target'] - avg_hold) / penalty_settings['hold']['target']
        else:
            penalty_hold = 1.0
        
        # --- Simulate trades to compute average pip gains and losses ---
        # Identify trade boundaries:
        change_indices = np.where(np.diff(np.sign(pos)) != 0)[0] + 1
        trade_starts = np.concatenate(([0], change_indices))
        trade_ends = np.concatenate((change_indices, [n_timesteps]))
        
        win_pips = []
        loss_pips = []
        for start, end in zip(trade_starts, trade_ends):
            if pos[start] == 0:
                continue
            # Sum log returns over the trade
            trade_log_return = np.sum(logr[start:end])
            # Convert log return to return (approximation, valid for small returns)
            trade_return = np.exp(trade_log_return) - 1
            # Compute pip difference:
            pips = trade_return / pip_size
            if pips > 0:
                win_pips.append(pips)
            else:
                loss_pips.append(abs(pips))
        
        # Average pip gains for wins, average pip losses for losses:
        avg_win = np.mean(win_pips) if len(win_pips) > 0 else 0
        avg_loss = np.mean(loss_pips) if len(loss_pips) > 0 else 0
        
        if penalty_settings['win']['active'] and avg_win < penalty_settings['win']['target']:
            penalty_win = 1 + penalty_settings['win']['beta'] * (penalty_settings['win']['target'] - avg_win) / penalty_settings['win']['target']
        else:
            penalty_win = 1.0
        
        if penalty_settings['loss']['active'] and avg_loss < penalty_settings['loss']['target']:
            penalty_loss = 1 + penalty_settings['loss']['beta'] * (penalty_settings['loss']['target'] - avg_loss) / penalty_settings['loss']['target']
        else:
            penalty_loss = 1.0

        total_penalty = penalty_hold * penalty_win * penalty_loss
        penalties.append(total_penalty)
    
    penalties = np.array(penalties)
    adjusted_SSR = raw_SSR / penalties
    return adjusted_SSR

def select_mating_pool(pop, fitness, num_parents):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    parents = np.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        max_fitness_idx = np.where(fitness == np.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = -99999999999
    return parents

def crossover(parents, offspring_size):
    offspring = np.empty(offspring_size)
    # The point at which crossover takes place between two parents. Usually, it is at the center.
    crossover_point = np.uint8(offspring_size[1]/2)

    for k in range(offspring_size[0]):
        # Index of the first parent to mate.
        parent1_idx = k%parents.shape[0]
        # Index of the second parent to mate.
        parent2_idx = (k+1)%parents.shape[0]
        # The new offspring will have its first half of its genes taken from the first parent.
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # The new offspring will have its second half of its genes taken from the second parent.
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring

def mutation(offspring_crossover, num_mutations=1):
    mutations_counter = np.uint8(offspring_crossover.shape[1] / num_mutations)
    # Mutation changes a number of genes as defined by the num_mutations argument. The changes are random.
    for idx in range(offspring_crossover.shape[0]):
        gene_idx = mutations_counter - 1
        for _ in range(num_mutations):
            # The random value to be added to the gene.
            random_value = np.random.uniform(-1.0, 1.0, 1)
            offspring_crossover[idx, gene_idx] = offspring_crossover[idx, gene_idx] + random_value
            gene_idx = gene_idx + mutations_counter
    return offspring_crossover

def GA_train(training_df, optimizing_selection=0, sol_per_pop=8, num_parents_mating=4, num_generations=200,
             penalty_mode=False, penalty_settings=None, print_interval=10):
    """
    Runs the Genetic Algorithm for a fixed number of generations to optimize the weight vector.
    
    Additional parameters:
      - penalty_mode: Boolean; if True, uses cal_pop_fitness with additional penalties.
      - penalty_settings: Dictionary of penalty settings.
      - print_interval: Number of generations between printing progress metrics.
    
    Returns:
      - The best candidate weight vector from the final generation.
    """
    # ANSI color codes for console output
    BOLD_GREEN  = "\033[1;32m"
    BOLD_CYAN   = "\033[1;36m"
    BOLD_YELLOW = "\033[1;33m"
    RESET       = "\033[0m"
    
    equation_inputs = training_df.values
    num_weights = training_df.shape[1] - 1
    pop_size = (sol_per_pop, num_weights)
    new_population = np.random.uniform(low=-1.0, high=1.0, size=pop_size)
    
    # Extract log returns for later calculations
    logr = np.asarray(equation_inputs[:, 0]).flatten()
    
    best_outputs = []
    best_weights_history = []
    
    print(f"{BOLD_YELLOW}Starting Genetic Algorithm Training with {num_generations} generations{RESET}")
    print(f"{BOLD_YELLOW}Population size: {sol_per_pop}, Number of parents: {num_parents_mating}{RESET}")
    
    for generation in range(num_generations):
        fitness = cal_pop_fitness(equation_inputs, new_population, optimizing_selection,
                                  penalty_mode, penalty_settings)
        best_fitness = np.max(fitness)
        best_outputs.append(best_fitness)
        
        # Find the best individual in this generation
        best_idx = np.argmax(fitness)
        best_weights = new_population[best_idx]
        best_weights_history.append(best_weights)
        
        # Calculate performance metrics for the best individual
        if generation % print_interval == 0 or generation == num_generations - 1:
            # Calculate position and returns
            position = np.dot(best_weights, equation_inputs[:, 1:].T)
            # Ensure position is properly shaped
            position = np.asarray(position).flatten()
            # Avoid division by zero
            max_abs_pos = np.max(np.abs(position))
            if max_abs_pos > 0:
                position = position / max_abs_pos  # Normalize positions
            # Ensure logr and position have the same shape for multiplication
            port_r = (position * logr).astype(np.float64)
            
            # Calculate key metrics
            cumulative_return = np.sum(port_r) * 100  # convert to percentage
            
            # Properly calculate Sharpe ratio (annualized assuming 5min data: 252*24*12 periods per year)
            sharpe = port_r.mean() / port_r.std() if port_r.std() != 0 else 0
            annual_factor = (252 * 24 * 12) ** 0.5
            annualized_sharpe = sharpe * annual_factor
            
            # Calculate win rate
            num_wins = np.sum(port_r > 0)
            win_rate = num_wins / len(port_r) * 100 if len(port_r) > 0 else 0
            
            # Calculate maximum drawdown
            cumulative = np.cumsum(port_r)
            max_dd = 0
            peak = cumulative[0]
            for value in cumulative:
                if value > peak:
                    peak = value
                dd = (peak - value) * 100  # convert to percentage
                if dd > max_dd:
                    max_dd = dd
            
            # Print progress
            print(f"{BOLD_GREEN}Generation {generation}/{num_generations} ({generation/num_generations*100:.1f}%){RESET}")
            print(f"{BOLD_CYAN}Best Fitness (SSR):{RESET} {best_fitness:.6f}")
            print(f"{BOLD_CYAN}Cumulative Return:{RESET} {cumulative_return:.2f}%")
            print(f"{BOLD_CYAN}Annualized Sharpe:{RESET} {annualized_sharpe:.3f}")
            print(f"{BOLD_CYAN}Win Rate:{RESET} {win_rate:.2f}%")
            print(f"{BOLD_CYAN}Max Drawdown:{RESET} {max_dd:.2f}%")
            print("-" * 50)
            
        # Select parents for next generation
        parents = select_mating_pool(new_population, fitness, num_parents_mating)
        offspring_crossover = crossover(parents, offspring_size=(pop_size[0]-parents.shape[0], num_weights))
        offspring_mutation = mutation(offspring_crossover, num_mutations=2)
        new_population[0:parents.shape[0], :] = parents
        new_population[parents.shape[0]:, :] = offspring_mutation

    # Final evaluation
    fitness = cal_pop_fitness(equation_inputs, new_population, optimizing_selection,
                              penalty_mode, penalty_settings)
    best_match_idx = np.where(fitness == np.max(fitness))
    
    # Plot progress of fitness over generations
    plt.figure(figsize=(10, 6))
    plt.plot(best_outputs)
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness (SSR ratio)")
    plt.title("Genetic Algorithm Progress")
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Find index of overall best performer across all generations
    overall_best_gen = np.argmax(best_outputs)
    print(f"{BOLD_YELLOW}Training complete! Best fitness achieved in generation {overall_best_gen}{RESET}")
    
    return new_population[best_match_idx]