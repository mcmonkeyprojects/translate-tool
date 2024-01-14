# mcmonkey's AI Translation Tool

Bulk translates everything in a reference file using local translation AI.

Built using https://github.com/huggingface/candle and by default uses this model: https://huggingface.co/jbochi/madlad400-7b-mt-bt in GGUF-q4 which is derived from https://huggingface.co/google/madlad400-7b-mt-bt

## Usage

- Install rust: https://www.rust-lang.org/tools/install
    - Expect silly rust errors that you have to google (eg requiring Visual Studio on Windows for some reason)

Compile:
```sh
cargo build --release
```

Run:
```sh
./target/release/translate-tool.exe --in-json "data/test-in.json" --out-json "data/test-out.json" --language de
```

Tack `--verbose` onto the end to get some live debug output as it goes.

Use `--model-id jbochi/madlad400-3b-mt` if you're impatient and want a smaller model.

On an Intel i7-12700KF, 7b-mt-bt runs at around 1 token/s, 3b-mt runs at around 2.8 token/s.

Example input JSON file:
```json
{
    "keys": {
        "This keys needs translation": "",
        "This key doesn't": "cause it has a value"
    }
}
```

This will translate keys and store the result in the value, skipping any keys that already have a value.

Language should be a standard language code - if in doubt, see list at https://arxiv.org/pdf/2309.04662.pdf Appendix A.1

Note that this runs entirely on CPU, because the Transformers GPU version needs too much VRAM to work and GGUF doesn't want to work on GPU within candle I guess? "Oh but why not use regular GGML to run it then" because GGML doesn't support T5??? Idk why candle supports GGML-formatted T5 but GGML itself doesn't. AI tech is a mess. If you're reading this after year 2024 when this was made there's hopefully less dumb ways to do what is currently cutting edge AI stuff.

This will burn your CPU and take forever.

Note that I'm not experienced in Rust and the lifetime syntax is painful so I might've screwed something up.

## Legal Stuff

This project depends on Candle which is either MIT or Apache2. Both licenses are in their repo don't ask me what that means idek.

Sections of source code are copied from Candle examples.

This project depends on MADLAD models that google research released under Apache2 which I'm not entirely clear why a software license is on model weights but again idek.

Anything unique to this project is yeeted out freely under the MIT license.

I have no idea whether any legal restrictions apply to the resultant translated text but you're probably fine probably (if you have rights to use the source text at least)

## License

The MIT License (MIT)

Copyright (c) 2024 Alex "mcmonkey" Goodwin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
