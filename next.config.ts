import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
    async redirects() {
      return [
        {
          source: '/',
          destination: '/accueil',
          permanent: false, // set to true if this is a permanent redirect (301)
        },
      ]
    },
  
};

export default nextConfig;
