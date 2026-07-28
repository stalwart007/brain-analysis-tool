import { cookies } from "next/headers";
import { redirect } from "next/navigation";
import Shell from "@/components/Shell";
import { SESSION_COOKIE, verifySessionToken } from "@/lib/auth";

export default async function AppLayout({ children }: { children: React.ReactNode }) {
  // Defense in depth: middleware already gates, but re-verify server-side.
  const token = (await cookies()).get(SESSION_COOKIE)?.value;
  const email = token ? await verifySessionToken(token) : null;
  if (!email) redirect("/login");

  // No separate backdrop: the cortex navigator inside Shell *is* the backdrop.
  return <Shell>{children}</Shell>;
}
