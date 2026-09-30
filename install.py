    # IMPORTANT: existing ComfyUI-GGUF is updated instead of skipped.
    git_clone_or_update(
        GGUF_REPO,
        gguf_dir,
        "ComfyUI-GGUF",
    )

    pip_requirements(
        gguf_dir,
        "ComfyUI-GGUF",
    )

    verify_qwen_image21_gguf(gguf_dir)

    # --------------------------------------------------------
    # DOWNLOAD ENGINE
    # --------------------------------------------------------
    section("DOWNLOAD ENGINE")

    use_aria2 = ensure_aria2(env)

    # --------------------------------------------------------
    # MODELS
    # --------------------------------------------------------
    section("MODEL ACQUISITION")

    model_paths = []

    for label, url, sub, fname in MODELS:
        dest = comfy_dir / "models" / sub

        download(
            label,
            url,
            dest,
            fname,
            use_aria2,
            token,
        )

        model_paths.append(
            (label, dest / fname)
        )

    # --------------------------------------------------------
    # VERIFY
    # --------------------------------------------------------
    missing = [
        p
        for _, p in model_paths
        if not p.exists()
    ]

    if missing:
        fail(
            "Missing files: "
            + ", ".join(p.name for p in missing)
        )
        return 1

    # --------------------------------------------------------
    # DONE
    # --------------------------------------------------------
    final_report(
        env,
        comfy_dir,
        model_paths,
        time.time() - T0,
        clear=not args.no_clear,
    )

    if args.launch:
        print(
            f"{C.CY}>> launching ComfyUI on port "
            f"{args.port} ...{C.X}\n"
        )

        os.chdir(comfy_dir)

        os.execv(
            sys.executable,
            [
                sys.executable,
                "main.py",
                "--listen",
                "0.0.0.0",
                "--port",
                str(args.port),
            ],
        )

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())

    except KeyboardInterrupt:
        print(
            f"\n{C.YL}[ABORT] interrupted by user{C.X}"
        )
        sys.exit(130)

    except Exception as e:
        print(f"\n{C.RD}[FATAL] {e}{C.X}")
        print(
            f"{C.DM}See {LOG_FILE.resolve()} "
            f"for details{C.X}"
        )
        sys.exit(1)
