# Strategic Reward Improvements for Balatro Agent

## Problem Analysis

The original agent was making suboptimal decisions, particularly in the example hand:
```
[2♣, J♠, 9♠, Q♦, K♥, 8♠, 7♠, 3♦]
```

**What the agent did:** Played a weak high card hand `[2♣, J♠, K♥, 8♠]` for only 35 points.

**What it should have done:** Discard the non-spade cards `[2♣, Q♦, K♥, 3♦]` to keep the spades `[J♠, 9♠, 8♠, 7♠]` and try for a flush.

## Root Cause

The original reward function had several issues:

1. **Weak penalties for playing bad hands** - Only -15 penalty for playing weak hands when discards were available
2. **Insufficient discard incentives** - Base discard reward was only 8.0
3. **No strategic awareness** - Didn't recognize flush potential or hand improvement opportunities
4. **Poor hand quality assessment** - Didn't consider strategic potential, only current hand strength

## Strategic Improvements Made

### 1. Enhanced Play Penalties

**Before:**
```python
if hand_type in ["High Card", "Pair"] and score_gained < 15 and self.discards_left > 0:
    reward -= 15.0
```

**After:**
```python
if hand_type in ["High Card", "Pair"] and score_gained < 25 and self.discards_left > 0:
    reward -= 25.0  # Much stronger penalty
```

**Impact:** Strongly discourages playing weak hands when better options exist.

### 2. Improved Discard Rewards

**Before:**
```python
reward = 8.0  # Base discard reward
```

**After:**
```python
reward = 12.0  # Increased base reward
# Plus strategic bonuses for good discarding
```

**Impact:** Makes discarding more attractive as a general strategy.

### 3. Strategic Discard Bonuses

**New Feature:** `_calculate_discard_strategy_bonus()`

```python
def _calculate_discard_strategy_bonus(self) -> float:
    # Count cards by suit
    suit_counts = {'H': 0, 'D': 0, 'C': 0, 'S': 0}
    for card in self.hand:
        suit_counts[card.suit] += 1
    
    # Find most common suit
    max_suit = max(suit_counts, key=suit_counts.get)
    max_count = suit_counts[max_suit]
    
    bonus = 0.0
    
    # Flush potential bonuses
    if max_count >= 3:
        bonus += 15.0  # Strong bonus for preserving flush potential
        other_suit_cards = sum(count for suit, count in suit_counts.items() if suit != max_suit)
        if other_suit_cards > 0:
            bonus += 5.0 * other_suit_cards  # Bonus per card of other suits
    elif max_count == 2:
        bonus += 8.0  # Moderate bonus for preserving pair potential
    
    return bonus
```

**Impact:** Specifically rewards discarding that improves hand potential (like going for flushes).

### 4. Enhanced Hand Quality Assessment

**New Feature:** `_calculate_hand_potential_bonus()`

```python
def _calculate_hand_potential_bonus(self) -> float:
    # Flush potential
    max_suit_count = max(suit_counts.values())
    if max_suit_count >= 4:
        bonus += 50.0  # Very strong flush potential
    elif max_suit_count == 3:
        bonus += 25.0  # Good flush potential
    elif max_suit_count == 2:
        bonus += 10.0  # Some flush potential
    
    # Pair/three-of-a-kind potential
    for rank, count in rank_counts.items():
        if count >= 3:
            bonus += 30.0  # Three of a kind potential
        elif count == 2:
            bonus += 15.0  # Pair potential
```

**Impact:** Better state representation that includes strategic potential, not just current strength.

## Expected Behavioral Changes

### For the Example Hand `[2♣, J♠, 9♠, Q♦, K♥, 8♠, 7♠, 3♦]`

**Before the improvements:**
- Agent sees 4 spades but doesn't recognize flush potential
- Plays weak high card hand for immediate small reward
- Gets -15 penalty, but still positive net reward

**After the improvements:**
- Agent recognizes 4 spades = strong flush potential (+50 bonus)
- Discarding non-spade cards gets strategic bonus (+15 base + 5×4 other cards = +35)
- Playing weak hand gets -25 penalty
- Net reward for discarding: ~+47 vs playing: ~-20
- **Expected behavior:** Agent will choose to discard

### General Strategic Improvements

1. **Flush Recognition:** Agent will now recognize and pursue flush opportunities
2. **Strategic Discarding:** Agent will discard cards that don't help the best potential hand
3. **Weak Hand Avoidance:** Agent will avoid playing weak hands when better options exist
4. **Hand Building:** Agent will build toward strong hands rather than playing mediocre ones immediately

## Training Recommendations

### 1. Retrain with New Rewards

```bash
python train_ppo.py --timesteps 50000 --blind-score 300
```

### 2. Monitor Strategic Behavior

Look for these indicators of improvement:
- Higher discard rates when flush potential exists
- Lower rates of playing weak high card hands
- Better final scores and win rates
- More strategic hand building

### 3. Fine-tuning Parameters

If the agent becomes too conservative:
- Reduce the play penalty from -25 to -20
- Increase the base discard reward from 12 to 15

If the agent still plays weak hands too often:
- Increase the play penalty further
- Add additional penalties for specific weak hand types

## Testing the Improvements

Run the test script to see the improvements in action:

```bash
python test_strategic_rewards.py
```

This will:
1. Analyze the example hand and show the optimal strategy
2. Demonstrate the new reward calculations
3. Show how the agent behaves with the improved rewards

## Expected Results

With these improvements, the agent should:

1. **Recognize flush opportunities** and discard non-matching cards
2. **Build toward strong hands** rather than playing weak ones immediately
3. **Make more strategic decisions** about when to play vs. discard
4. **Achieve higher scores** and better win rates
5. **Play more like a skilled human player** who understands hand potential

The key insight is that Balatro rewards strategic thinking and hand building, not just immediate point scoring. These improvements align the agent's incentives with optimal play patterns. 