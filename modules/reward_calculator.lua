local RewardCalculator = {}

-- Initialize reward tracking
function RewardCalculator.init()
    RewardCalculator.cumulative_reward = 0
    RewardCalculator.best_cumulative_reward = -math.huge
    RewardCalculator.current_run_reward = 0
    RewardCalculator.last_chips = 0
    RewardCalculator.saves_count = 0
end

-- Calculate reward based on game state changes
function RewardCalculator.calculate()
    local reward = 0
    local done = false

    -- Base reward for winning with a hand
    if G.GAME.chips > G.GAME.blind.chips then
        reward = reward + 10
        print("Added +10 reward for winning hand")
    end

    -- Reward for chips gained
    local chip_difference = G.GAME.chips - RewardCalculator.last_chips
    reward = reward + chip_difference * 0.1
    print("Chip difference: " .. tostring(chip_difference))
    print("Added reward from chips: " .. tostring(chip_difference * 0.1))

    -- Penalty for game over
    if G.STATE == G.STATES.GAME_OVER then
        reward = reward - 20
        done = true
        print("Added -20 penalty for game over")
    end

    return reward, done
end

-- Update reward tracking
function RewardCalculator.updateTracking(reward, is_run_end)
    RewardCalculator.cumulative_reward = RewardCalculator.cumulative_reward + reward
    RewardCalculator.current_run_reward = RewardCalculator.current_run_reward + reward

    print(string.format("Reward: %.2f, Current Run Total: %.2f, Cumulative: %.2f",
        reward, RewardCalculator.current_run_reward, RewardCalculator.cumulative_reward))

    -- Reset current run reward if run ended
    if is_run_end then
        print(string.format("Run ended. Run reward: %.2f, Cumulative: %.2f",
            RewardCalculator.current_run_reward, RewardCalculator.cumulative_reward))
        RewardCalculator.current_run_reward = 0
    end
end

-- Check if we achieved a new best cumulative reward
function RewardCalculator.checkNewBest()
    if RewardCalculator.cumulative_reward > RewardCalculator.best_cumulative_reward then
        local old_best = RewardCalculator.best_cumulative_reward
        RewardCalculator.best_cumulative_reward = RewardCalculator.cumulative_reward
        print(string.format("New best cumulative reward! Old: %.2f, New: %.2f", 
            old_best, RewardCalculator.best_cumulative_reward))
        return true
    end
    return false
end

-- Store pre-action state for reward calculation
function RewardCalculator.storePreActionState()
    RewardCalculator.last_chips = G.GAME.chips
    print("Stored pre-action chips: " .. RewardCalculator.last_chips)
end

-- Reset rewards for new game session
function RewardCalculator.resetForNewSession()
    RewardCalculator.cumulative_reward = 0
    RewardCalculator.current_run_reward = 0
    print("Reset cumulative rewards for new game session")
end

-- Reset rewards for new run after game over
function RewardCalculator.resetForNewRun()
    RewardCalculator.cumulative_reward = 0
    RewardCalculator.current_run_reward = 0
    print("Reset cumulative rewards for new run after game over")
end

-- Get reward statistics
function RewardCalculator.getStats()
    return {
        cumulative_reward = RewardCalculator.cumulative_reward,
        best_cumulative_reward = RewardCalculator.best_cumulative_reward,
        current_run_reward = RewardCalculator.current_run_reward,
        saves_count = RewardCalculator.saves_count
    }
end

-- Increment save count
function RewardCalculator.incrementSaveCount()
    RewardCalculator.saves_count = RewardCalculator.saves_count + 1
    print("Model saved successfully! Save count: " .. RewardCalculator.saves_count)
    print("Saved at cumulative reward: " .. RewardCalculator.cumulative_reward)
end

return RewardCalculator 