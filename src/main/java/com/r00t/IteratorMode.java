package com.r00t;

public enum IteratorMode {
    TRAINING("train"),
    TESTING("test");

    private final String s;

    IteratorMode(String s) {
        this.s = s;
    }

    public String getPath() {
        return s;
    }
}
