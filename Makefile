PREFIX ?= /usr/local
BINARY = apfel

.PHONY: build install uninstall clean version-bump reinstall all

all: reinstall

build:
	swift build -c release

install: build
	@if [ -w "$(PREFIX)/bin" ]; then \
		install .build/release/$(BINARY) $(PREFIX)/bin/$(BINARY); \
	else \
		sudo install .build/release/$(BINARY) $(PREFIX)/bin/$(BINARY); \
	fi
	@echo "✓ installed: $$($(PREFIX)/bin/$(BINARY) --version)"

reinstall:
	@pkill -f "apfel --serve" 2>/dev/null || true
	@pkill -f "apfel --gui" 2>/dev/null || true
	@sleep 1
	swift build -c release
	@if [ -w "$(PREFIX)/bin" ]; then \
		install .build/release/$(BINARY) $(PREFIX)/bin/$(BINARY); \
	else \
		sudo install .build/release/$(BINARY) $(PREFIX)/bin/$(BINARY); \
	fi
	@echo "✓ installed: $$($(PREFIX)/bin/$(BINARY) --version)"

uninstall:
	@if [ -w "$(PREFIX)/bin" ]; then \
		rm -f $(PREFIX)/bin/$(BINARY); \
	else \
		sudo rm -f $(PREFIX)/bin/$(BINARY); \
	fi

clean:
	swift package clean

version-bump:
	@v=$$(grep 'let version = ' Sources/main.swift | head -1 | sed 's/.*"\(.*\)"/\1/'); \
	major=$$(echo $$v | cut -d. -f1); \
	minor=$$(echo $$v | cut -d. -f2); \
	patch=$$(echo $$v | cut -d. -f3); \
	new="$$major.$$minor.$$((patch+1))"; \
	sed -i '' "s/let version = \"$$v\"/let version = \"$$new\"/" Sources/main.swift; \
	echo "$$v → $$new"
