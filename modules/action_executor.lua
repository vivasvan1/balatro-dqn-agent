local ActionExecutor = {}

-- Helper function to select cards by indices
function ActionExecutor.selectCardsByIndices(indices)
    if not G or not G.hand or not G.hand.cards then
        print("ERROR: No valid hand found.")
        return false
    end

    -- First unhighlight all cards
    G.hand:unhighlight_all()

    -- Validate indices
    for _, idx in ipairs(indices) do
        if type(idx) ~= "number" or idx < 1 or idx > #G.hand.cards then
            print("ERROR: Invalid index: " .. tostring(idx) .. " (hand size: " .. #G.hand.cards .. ")")
            return false
        end
    end

    -- Highlight cards at specified indices
    for _, idx in ipairs(indices) do
        local card = G.hand.cards[idx]
        card:highlight(true)
        table.insert(G.hand.highlighted, card)
        print(string.format("Selected card %d: %s", idx, card.base.value .. card.base.suit))
    end

    -- If we're selecting a hand, make sure to parse the highlighted cards
    if G.STATE == G.STATES.SELECTING_HAND then
        G.hand:parse_highlighted()
    end

    print("Highlighted " .. #indices .. " cards in hand.")
    return true
end

-- Helper function to select random hand cards
function ActionExecutor.selectRandomCards()
    if not G or not G.hand or not G.hand.cards or #G.hand.cards < 5 then
        print("ERROR: Not enough cards in hand to select 5.")
        return false
    end

    -- Generate random indices
    local indices = {}
    for i = 1, #G.hand.cards do 
        table.insert(indices, i) 
    end
    
    -- Shuffle indices
    for i = #indices, 2, -1 do
        local j = math.random(i)
        indices[i], indices[j] = indices[j], indices[i]
    end

    -- Select first 5 random cards
    local selected_indices = {}
    local max_cards = math.min(5, #G.hand.cards)
    for i = 1, max_cards do
        table.insert(selected_indices, indices[i])
    end

    return ActionExecutor.selectCardsByIndices(selected_indices)
end

-- Execute an action based on action data from server
function ActionExecutor.execute(action_data)
    if not action_data or not action_data.indices or not action_data.action then
        print("ERROR: Invalid action data received: 'action' or 'indices' field missing")
        if action_data then
            print("Received action_data.action: " .. tostring(action_data.action))
            print("Received action_data.indices: " .. (action_data.indices and "table" or "nil"))
        end
        return false
    end

    -- Validate we're in the right state
    if G.STATE ~= G.STATES.SELECTING_HAND then
        print("ERROR: Cannot execute action, not in SELECTING_HAND state. Current: " .. tostring(G.STATE))
        return false
    end

    -- Select the cards first
    if not ActionExecutor.selectCardsByIndices(action_data.indices) then
        print("ERROR: Failed to select cards for action")
        return false
    end

    -- Execute the action
    if action_data.action == "play" then
        print("Executing: Playing selected cards")
        G.FUNCS.play_cards_from_highlighted()
        return true
    elseif action_data.action == "discard" then
        print("Executing: Discarding selected cards")
        G.FUNCS.discard_cards_from_highlighted()
        return true
    else
        print("ERROR: Unknown action type: " .. tostring(action_data.action))
        return false
    end
end

-- Validate action data structure
function ActionExecutor.validateActionData(action_data)
    if not action_data then
        return false, "Action data is nil"
    end
    
    if not action_data.action then
        return false, "Missing 'action' field"
    end
    
    if not action_data.indices then
        return false, "Missing 'indices' field"
    end
    
    if type(action_data.indices) ~= "table" then
        return false, "'indices' must be a table"
    end
    
    if #action_data.indices == 0 then
        return false, "'indices' table is empty"
    end
    
    if action_data.action ~= "play" and action_data.action ~= "discard" then
        return false, "Invalid action type: " .. tostring(action_data.action)
    end
    
    return true, "Valid"
end

return ActionExecutor 