local StateMachine = {}

-- Agent phases
StateMachine.PHASES = {
    IDLE = "IDLE",
    SELECTING_BLIND = "SELECTING_BLIND",
    SELECTING_HAND = "SELECTING_HAND",
    ACTION_PENDING = "ACTION_PENDING",
    WAITING_FOR_REWARD = "WAITING_FOR_REWARD",
    PROCESSING_REWARD = "PROCESSING_REWARD",
    GAME_OVER = "GAME_OVER"
}

-- Initialize state machine
function StateMachine.init()
    StateMachine.current_phase = StateMachine.PHASES.IDLE
    StateMachine.previous_phase = nil
    StateMachine.phase_start_time = love.timer.getTime()
    StateMachine.callbacks = {}
end

-- Register callback for phase transitions
function StateMachine.onPhaseChange(callback)
    table.insert(StateMachine.callbacks, callback)
end

-- Change phase with logging and callbacks
function StateMachine.setPhase(new_phase, reason)
    if StateMachine.current_phase ~= new_phase then
        local old_phase = StateMachine.current_phase
        StateMachine.previous_phase = old_phase
        StateMachine.current_phase = new_phase
        StateMachine.phase_start_time = love.timer.getTime()
        
        print(string.format("State Machine: %s -> %s. Reason: %s", 
            old_phase, new_phase, reason or "N/A"))
        
        -- Call registered callbacks
        for _, callback in ipairs(StateMachine.callbacks) do
            local success, err = pcall(callback, old_phase, new_phase, reason)
            if not success then
                print("ERROR in phase change callback: " .. tostring(err))
            end
        end
    end
end

-- Get current phase
function StateMachine.getPhase()
    return StateMachine.current_phase
end

-- Get previous phase
function StateMachine.getPreviousPhase()
    return StateMachine.previous_phase
end

-- Get time spent in current phase
function StateMachine.getPhaseTime()
    return love.timer.getTime() - StateMachine.phase_start_time
end

-- Check if in specific phase
function StateMachine.isInPhase(phase)
    return StateMachine.current_phase == phase
end

-- Check if transitioning is allowed
function StateMachine.canTransitionTo(target_phase)
    local current = StateMachine.current_phase
    local valid_transitions = {
        [StateMachine.PHASES.IDLE] = {
            StateMachine.PHASES.SELECTING_BLIND,
            StateMachine.PHASES.GAME_OVER
        },
        [StateMachine.PHASES.SELECTING_BLIND] = {
            StateMachine.PHASES.SELECTING_HAND,
            StateMachine.PHASES.GAME_OVER
        },
        [StateMachine.PHASES.SELECTING_HAND] = {
            StateMachine.PHASES.ACTION_PENDING,
            StateMachine.PHASES.GAME_OVER
        },
        [StateMachine.PHASES.ACTION_PENDING] = {
            StateMachine.PHASES.WAITING_FOR_REWARD,
            StateMachine.PHASES.IDLE,
            StateMachine.PHASES.GAME_OVER
        },
        [StateMachine.PHASES.WAITING_FOR_REWARD] = {
            StateMachine.PHASES.PROCESSING_REWARD,
            StateMachine.PHASES.GAME_OVER
        },
        [StateMachine.PHASES.PROCESSING_REWARD] = {
            StateMachine.PHASES.SELECTING_HAND,
            StateMachine.PHASES.GAME_OVER,
            StateMachine.PHASES.IDLE
        },
        [StateMachine.PHASES.GAME_OVER] = {
            StateMachine.PHASES.IDLE,
            StateMachine.PHASES.SELECTING_BLIND
        }
    }
    
    local allowed = valid_transitions[current]
    if not allowed then
        return false
    end
    
    for _, allowed_phase in ipairs(allowed) do
        if allowed_phase == target_phase then
            return true
        end
    end
    
    return false
end

-- Safe phase transition with validation
function StateMachine.transitionTo(target_phase, reason)
    if not StateMachine.canTransitionTo(target_phase) then
        print(string.format("WARNING: Invalid transition from %s to %s. Reason: %s",
            StateMachine.current_phase, target_phase, reason or "N/A"))
        return false
    end
    
    StateMachine.setPhase(target_phase, reason)
    return true
end

-- Reset state machine
function StateMachine.reset()
    StateMachine.setPhase(StateMachine.PHASES.IDLE, "State machine reset")
end

-- Get state machine status for debugging
function StateMachine.getStatus()
    return {
        current_phase = StateMachine.current_phase,
        previous_phase = StateMachine.previous_phase,
        phase_time = StateMachine.getPhaseTime(),
        phase_start_time = StateMachine.phase_start_time
    }
end

return StateMachine 