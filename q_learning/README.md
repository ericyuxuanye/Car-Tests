# Q Learning

This implements a basic Deep Q Learning model, using a discrete action space. To
train the model, make sure to set the variable in `LIVE` in `game.py` to
`False`, and run the game using

```sh
python game.py
```

To improve performance, no game is being rendered in this mode while the AI
trains. You can exit any time by initiating a `SIGINT` signal, and the data
will be saved immediately before the program exits.

If you want to see the model in action, set the variable `LIVE` in `game.py` to
`True`.
