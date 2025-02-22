import Intro from '@/app/ui/accueil/intro';
import Navbar from '../ui/accueil/navbar'; 

export default function Layout({ children }: { children: React.ReactNode }) {
  return (
    <div>
        <div>
            <Navbar />
        </div>
      <div className="md:overflow-y-auto h-screen ">{children}</div>
    </div>
  );
}