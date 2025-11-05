If you want to be able to just say:

```sh
./llava.llamafile
```

...and have it run the web server without having to specify arguments,
then you can embed both the weights and a special `.args` inside, which
specifies the default arguments. First, let's create a file named
`.args` which has this content:

```sh
-m
llava-v1.5-7b-Q8_0.gguf
--mmproj
llava-v1.5-7b-mmproj-Q8_0.gguf
--host
0.0.0.0
-ngl
9999
...
```

As we can see above, there's one argument per line. The `...` argument
optionally specifies where any additional CLI arguments passed by the
user are to be inserted. Next, we'll add both the weights and the
argument file to the executable:

```sh
cp /usr/local/bin/llamafile llava.llamafile

zipalign -j0 \
  llava.llamafile \
  llava-v1.5-7b-Q8_0.gguf \
  llava-v1.5-7b-mmproj-Q8_0.gguf \
  .args

./llava.llamafile
```

Congratulations. You've just made your own LLM executable that's easy to
share with your friends.


## Distribution

One good way to share a llamafile with your friends is by posting it on
Hugging Face. If you do that, then it's recommended that you mention in
your Hugging Face commit message what git revision or released version
of llamafile you used when building your llamafile. That way everyone
online will be able verify the provenance of its executable content. If
you've made changes to the llama.cpp or cosmopolitan source code, then
the Apache 2.0 license requires you to explain what changed. One way you
can do that is by embedding a notice in your llamafile using `zipalign`
that describes the changes, and mention it in your Hugging Face commit.
