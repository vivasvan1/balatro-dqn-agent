local InputHandler = {}
local Utils = require("modules/utils")
local GameState = require("modules/game_state")

-- Input state tracking
local ctrl_key_pressed = false
local trigger_cooldown = 0.5
local last_trigger_time = nil

-- Initialize input handler
function InputHandler.init()
    ctrl_key_pressed = false
    last_trigger_time = nil
end

-- Check if ctrl+key combination is pressed
function InputHandler.checkCtrlKey(key)
    local ctrl = love.keyboard.isDown("lctrl") or love.keyboard.isDown("rctrl")
    local key_pressed = love.keyboard.isDown(key)

    if ctrl and key_pressed and not ctrl_key_pressed then
        print("Ctrl+" .. key .. " detected!")
    end

    if ctrl and key_pressed then
        if not ctrl_key_pressed then
            ctrl_key_pressed = true
            return true
        end
    else
        ctrl_key_pressed = false
    end
    return false
end

-- Handle Ctrl+G for manual action trigger
function InputHandler.handleCtrlG(callback)
    if InputHandler.checkCtrlKey("g") then
        local current_time = love.timer.getTime()
        
        if last_trigger_time == nil then
            last_trigger_time = current_time
        elseif current_time - last_trigger_time >= trigger_cooldown then
            print("Current time: " .. tostring(current_time))
            print("Last trigger time: " .. tostring(last_trigger_time))
            
            if callback then
                local success, result = pcall(callback)
                if not success then
                    print("ERROR in Ctrl+G callback: " .. tostring(result))
                end
            end
            
            last_trigger_time = current_time
        else
            print("Ctrl+G pressed: Cooldown active.")
        end
    end
end

-- Handle Ctrl+P for printing hand state
function InputHandler.handleCtrlP()
    if InputHandler.checkCtrlKey("p") then
        InputHandler.printHandState()
    end
end

-- Print current hand state
function InputHandler.printHandState()
    if not G or not G.hand or not G.hand.cards then
        print("ERROR: No valid hand found for printing")
        return
    end

    -- Print selected hand
    print("\nSelected Hand:")
    for _, card in ipairs(G.hand.highlighted) do
        print(string.format("Selected: %s", card.base.value .. card.base.suit))
    end

    -- Get and save hand state
    local hand_state = GameState.getHandState()
    
    -- Save hand state and blind info to JSON
    if Utils.saveGameStateToJson then
        Utils.saveGameStateToJson(hand_state, "hand_state.json")
        if G.GAME and G.GAME.blind then
            Utils.saveGameStateToJson(G.GAME.blind, "G.GAME.blind.json")
        end
    end
    
    print("Hand state saved to JSON files")
end

-- Handle Ctrl+S for saving game state
function InputHandler.handleCtrlS()
    if InputHandler.checkCtrlKey("s") then
        InputHandler.saveGameState()
    end
end

-- Save current game state to JSON
function InputHandler.saveGameState()
    if not Utils.saveGameStateToJson then
        print("ERROR: Utils.saveGameStateToJson not available")
        return
    end
    
    print("Saving game state...")
    
    -- Save various game states
    local states_to_save = {
        {data = GameState.process(), filename = "current_state.json"},
        {data = GameState.getStateInfo(), filename = "state_info.json"}
    }
    
    if G.GAME then
        table.insert(states_to_save, {data = G.GAME, filename = "G.GAME.json"})
    end
    
    for _, state in ipairs(states_to_save) do
        local success = Utils.saveGameStateToJson(state.data, state.filename)
        if success then
            print("Saved: " .. state.filename)
        else
            print("Failed to save: " .. state.filename)
        end
    end
end

-- Update input handler (call this in main update loop)
function InputHandler.update(manual_trigger_callback)
    InputHandler.handleCtrlG(manual_trigger_callback)
    InputHandler.handleCtrlP()
    InputHandler.handleCtrlS()
end

-- Get input handler status
function InputHandler.getStatus()
    return {
        ctrl_key_pressed = ctrl_key_pressed,
        last_trigger_time = last_trigger_time,
        trigger_cooldown = trigger_cooldown
    }
end

-- Set trigger cooldown
function InputHandler.setTriggerCooldown(cooldown)
    trigger_cooldown = cooldown
    print("Input trigger cooldown set to: " .. cooldown .. " seconds")
end

return InputHandler 