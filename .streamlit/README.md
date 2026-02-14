# Streamlit Configuration

This directory contains Streamlit-specific configuration files.

## Files

### config.toml
Main Streamlit configuration file that sets:
- Theme colors (blue primary color matching AdaptNav branding)
- Server settings (headless mode for deployment)
- Browser settings (disable usage stats)

## Customization

You can modify `config.toml` to change:
- **Theme colors**: primaryColor, backgroundColor, etc.
- **Server port**: Default is 8501
- **Browser behavior**: Auto-open, gather stats, etc.

See [Streamlit Configuration Docs](https://docs.streamlit.io/library/advanced-features/configuration) for more options.
