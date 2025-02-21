Thanks for attempting the challenges!

# Task E
In general, good attempt! I can also see yoo provided other implementations like a memory efficient Linear layer which is a fantastic addition!

1. But sadly, logits were not upcast to `float32` - this gets an instant 0 as per the guidelines sorry!
2. Mixed precision accumulation requires weights to be in `float32`, not `bfloat16`
3. There's also no test on Llama 1B training, so sadly again this is a must.

# Bugs Attempt
Unfortunately I cannot award any points, since this looks fully AI generated with no tests.
The first might be partially right though.
The bug bounties require an actual successful pull request.

### Thanks for the attempt again! Appreciate the effort!
