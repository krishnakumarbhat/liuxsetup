#!/bin/bash

# System Update Script for Pop!_OS / Debian-based systems

echo "🚀 Starting System Update..."

# Update package list and upgrade packages
echo "📦 Updating APT packages..."
sudo apt update && sudo apt upgrade -y

# Update Flatpaks if installed
if command -v flatpak &> /dev/null; then
    echo "📦 Updating Flatpaks..."
    flatpak update -y
else
    echo "⚠️ Flatpak not found, skipping..."
fi

# Cleanup
echo "🧹 Cleaning up..."
sudo apt autoremove -y
sudo apt autoclean

echo "✅ System Update Complete!"
