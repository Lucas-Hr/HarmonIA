import Intro from '@/app/ui/Home/intro';
import Navbar from '../ui/Home/navbar'; 

export default function Layout({ children }: { children: React.ReactNode }) {
  return (
    <div>
        <div>
            <Navbar />
        </div>
      <div className="md:overflow-y-auto h-screen bg-piano bg-cover">{children}</div>
    </div>
  );
}