export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ||
  (() => {
    if (typeof window === "undefined") return "http://localhost:9000";
    const hostname = window.location.hostname;
    const protocol = window.location.protocol;
    return `${protocol}//${hostname}:9000`;
  })();
